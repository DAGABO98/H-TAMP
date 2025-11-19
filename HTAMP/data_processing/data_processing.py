import argparse
from datetime import datetime
import re
import traceback

import numpy as np
import pandas as pd
import yaml

from HTAMP.data_processing.processing_dataclasses import HospitalDataFields, HospitalDataFiles

class DataProcessor:
    def __init__(self, hospital_data_files: HospitalDataFiles, hospital_config_file: str, hospital_data_fields: HospitalDataFields):
        self.hospital_data_files = hospital_data_files
        self.hospital_config_file = hospital_config_file
        self.hospital_data_fields = hospital_data_fields
        self.hospital_data = self._load_hospital_data()
        self.space_lookup = self._extract_space_lookup(hospital_config_file)
        self.stays_df = self._extract_patient_room_stays()
        self.medication_orders_df = self._annotate_medication_orders_with_room()
        
        self.medication_orders_df.to_csv("data/Annotated_Medications_Data.csv", index=False)

    def _load_hospital_data(self) -> dict[str, pd.DataFrame]:
        """Load and concatenate hospital data from multiple CSV files."""
        data_frames = {}
        for file_attr in vars(self.hospital_data_files):
            file_path = getattr(self.hospital_data_files, file_attr)
            df = pd.read_csv(file_path)
            df = df.drop_duplicates().reset_index(drop=True)
            data_frames[file_attr] = df

        return data_frames
    
    def _extract_space_lookup(self, file_path: str):
        space_config = yaml.safe_load(open(file_path, 'r'))
        space_lookup = {}
        for item in space_config.get("rooms", []):
            rid = item["id"]
            for loc in item["locations"]:
                space_lookup[loc] = rid
                    
        return space_lookup
    
    def _compose_location_string(self, department, room) -> str:
        department = str(department).strip()
        department = re.sub(r'\s+', '-', department)
        room = str(room).strip()
        room = re.sub(r'\s+', '-', room)
        return f"{department}_{room}"
    
    def _map_location_to_space(self, location: str) -> str:
        return self.space_lookup.get(location, "UNKNOWN_SPACE")
    
    def _extract_patient_room_stays(self) -> pd.DataFrame:
        """Extract patient room stay data."""
        visits_data = self.hospital_data['visits_data']
        visits_time_columns = self.hospital_data_fields.visits_time_in_columns + \
                              self.hospital_data_fields.visits_time_out_columns
        
        for c in [col for col in visits_time_columns if col in visits_data.columns]:
            visits_data[c] = pd.to_datetime(visits_data[c], errors='coerce')

        visits_data["IN_LOCATION"] = visits_data.apply(
            lambda row: self._compose_location_string(department=row[self.hospital_data_fields.visits_dept_in_column],
                                                     room=row[self.hospital_data_fields.visits_location_in_column]),
            axis=1
        )

        visits_data["OUT_LOCATION"] = visits_data.apply(
            lambda row: self._compose_location_string(department=row[self.hospital_data_fields.visits_dept_out_column],
                                                     room=row[self.hospital_data_fields.visits_location_out_column]),
            axis=1
        )

        # Build event stream (IN + OUT)
        in_events = (
            visits_data[self.hospital_data_fields.visits_time_in_columns + \
                        [self.hospital_data_fields.visits_patient_id_column, 
                         "IN_LOCATION"]]
            .rename(columns={"IN_LOCATION":"room", "IN_TIME":"ts"})
            .assign(etype="IN")
        )

        out_events = (
            visits_data[self.hospital_data_fields.visits_time_out_columns + \
                        [self.hospital_data_fields.visits_patient_id_column, 
                         "OUT_LOCATION"]]
            .rename(columns={"OUT_LOCATION":"room", "OUT_TIME":"ts"})
            .assign(etype="OUT")
        )

        events = pd.concat([in_events, out_events], ignore_index=True)
        events = events.dropna(subset=["ts", "room"])

        # Order so ties process OUT before IN
        events["etype_order"] = events["etype"].map({"OUT": 0, "IN": 1})

        # Compute stays: for each encounter, each IN lasts until the next event (or discharge)
        stays = []
        for key, g in events.groupby(self.hospital_data_fields.visits_patient_id_column, dropna=False):
            g = g.sort_values(["ts", "etype_order"], kind="mergesort")  # stable for ties

            adm = visits_data.loc[
                (visits_data[self.hospital_data_fields.visits_patient_id_column] == key),
                "HOSPITAL_ADMISSION"
            ].min()

            disch = visits_data.loc[
                (visits_data[self.hospital_data_fields.visits_patient_id_column] == key),
                "HOSPITAL_DISCHARGE"
            ].max()

            # --- NEW: synthesize an IN if the first event is OUT -------------------------
            if not g.empty and g.iloc[0]["etype"] == "OUT":
                first = g.iloc[0]
                start_ts = adm if pd.notna(adm) and adm <= first["ts"] else first["ts"]
                synthetic_in = {
                    **first.to_dict(),             # copy room, ids, discharge, etc.
                    "ts": start_ts,                # start at admission (or the OUT time as fallback)
                    "etype": "IN",
                    "etype_order": 1
                }
                g = pd.concat([pd.DataFrame([synthetic_in]), g], ignore_index=True)
                g = g.sort_values(["ts", "etype_order"], kind="mergesort")
            # ---------------------------------------------------------------------------

            g["next_ts"] = g["ts"].shift(-1)
            occ = g[g["etype"] == "IN"].copy()
            occ["end"] = occ["next_ts"].fillna(disch) 
            
            #occ = occ.dropna(subset=["end"])  # drop if we truly have no end
            occ["end"] = occ["end"].fillna(pd.Timestamp.now())

            occ["duration_minutes"] = (occ["end"] - occ["ts"]).dt.total_seconds() / 60.0
            occ = occ.rename(columns={"ts":"start", "room":"location"})
            stays.append(occ[[self.hospital_data_fields.visits_patient_id_column, 
                              "location", 
                              "start", 
                              "end", 
                              "duration_minutes"]])

        result = pd.concat(stays, ignore_index=True).sort_values([self.hospital_data_fields.visits_patient_id_column, "start"])

        result["space_id"] = result["location"].apply(self._map_location_to_space)

        return result
    
    def _annotate_events_with_room(self, 
                                  events_df: pd.DataFrame, 
                                  ts_col: str, 
                                  rooms_df: pd.DataFrame,
                                  new_room_label_col: str,
                                  new_space_id_col: str,
                                  new_start_col: str,
                                  new_end_col: str,
                                  patient_id_col: str) -> pd.DataFrame:
        r = rooms_df.copy()
        r[patient_id_col] = r[patient_id_col].astype(str)
        r["start"] = pd.to_datetime(r["start"], errors="coerce", utc=True).dt.tz_convert(None)
        r["end"]   = pd.to_datetime(r["end"],   errors="coerce", utc=True).dt.tz_convert(None)
        r["end"]   = r["end"].fillna(pd.Timestamp.max)
        r = r.sort_values(["start"], kind="mergesort").reset_index(drop=True)

        e = events_df.copy()
        e[patient_id_col] = e[patient_id_col].astype(str)
        e[ts_col] = pd.to_datetime(e[ts_col], errors="coerce", utc=True).dt.tz_convert(None)
        e = e.dropna(subset=[ts_col]).sort_values([ts_col], kind="mergesort").reset_index(drop=True)

        m = pd.merge_asof(
            e,
            r[[patient_id_col,"location","space_id","start","end"]],
            left_on=ts_col, right_on="start",
            by=patient_id_col,
            direction="backward",
            allow_exact_matches=True,
        )

        mask = m[ts_col].lt(m["end"])   # [start, end)
        m.loc[~mask, ["location","space_id","start","end"]] = [pd.NA, pd.NA, pd.NaT, pd.NaT]
        m.dropna(subset=["location"], inplace=True)
        return m.rename(columns={"location":new_room_label_col,"space_id":new_space_id_col,
                                "start":new_start_col,"end":new_end_col})
    
    def _annotate_medication_orders_with_room(self) -> pd.DataFrame:
        """Annotate medication orders with room stay information."""

        med_ordered_df = self._annotate_events_with_room(
            events_df=self.hospital_data['medications_orders'],
            ts_col="Medication Order DTTM",
            rooms_df=self.stays_df,
            new_room_label_col="ordered_room",
            new_space_id_col="ordered_space_id",
            new_start_col="ordered_start",
            new_end_col="ordered_end",
            patient_id_col=self.hospital_data_fields.visits_patient_id_column
        )

        scheduled_df = self._annotate_events_with_room(
            events_df=med_ordered_df,
            ts_col="Medication Scheduled DTTM",
            rooms_df=self.stays_df,
            new_room_label_col="scheduled_room",
            new_space_id_col="scheduled_space_id",
            new_start_col="scheduled_start",
            new_end_col="scheduled_end",
            patient_id_col=self.hospital_data_fields.visits_patient_id_column
        )

        administered_df = self._annotate_events_with_room(
            events_df=scheduled_df,
            ts_col="Administered DTTM",
            rooms_df=self.stays_df,
            new_room_label_col="administered_room",
            new_space_id_col="administered_space_id",
            new_start_col="administered_start",
            new_end_col="administered_end",
            patient_id_col=self.hospital_data_fields.visits_patient_id_column
        )

        administered_df.drop(columns=["ordered_start",
                                  "ordered_end",
                                  "scheduled_start",
                                  "scheduled_end",
                                  "administered_start", 
                                  "administered_end", 
                                  "Patient ID", 
                                  "Patient Encounter CSN", 
                                  "Race",
                                  "Age at Admission",
                                  "Order Med ID"], inplace=True)
        
        return administered_df


def main():
    parser = argparse.ArgumentParser(description="Process hospital data to extract patient room stays.")
    parser.add_argument("--visits_data_file", type=str, default="data/Visit_Data.csv", help="Path to the visits data CSV file.")
    parser.add_argument("--medications_orders_file", type=str, default="data/Medications_Data.csv", help="Path to the medications orders CSV file.")
    parser.add_argument("--blood_pressure_orders_file", type=str, default="data/Blood_Pressure_Data.csv", help="Path to the blood pressure orders CSV file.")
    parser.add_argument("--heart_rate_orders_file", type=str, default="data/Heart_Rate_Data.csv", help="Path to the heart rate orders CSV file.")
    parser.add_argument("--respiratory_rate_orders_file", type=str, default="data/Respiration_Data.csv", help="Path to the respiratory rate orders CSV file.")
    parser.add_argument("--temperature_orders_file", type=str, default="data/Temperature_Data.csv", help="Path to the temperature orders CSV file.")
    parser.add_argument("--oxygen_saturation_orders_file", type=str, default="data/SP02_Data.csv", help="Path to the oxygen saturation orders CSV file.")
    args = parser.parse_args()

    hospital_data_files = HospitalDataFiles(
        medications_orders=args.medications_orders_file,
        blood_pressure_orders=args.blood_pressure_orders_file,
        heart_rate_orders=args.heart_rate_orders_file,
        respiratory_rate_orders=args.respiratory_rate_orders_file,
        temperature_orders=args.temperature_orders_file,
        oxygen_saturation_orders=args.oxygen_saturation_orders_file,
        visits_data=args.visits_data_file
    )

    hospital_data_fields = HospitalDataFields(
        visits_patient_id_col="MRN",
        visits_location_in_col="IN_BED",
        visits_location_out_col="OUT_ROOM",
        visits_dept_in_col="IN_DEP",
        visits_dept_out_col="OUT_DEP",
        visits_time_in_cols=["IN_TIME", "HOSPITAL_ADMISSION"],
        visits_time_out_cols=["OUT_TIME", "HOSPITAL_DISCHARGE"]
    )

    processor = DataProcessor(hospital_data_files=hospital_data_files, 
                              hospital_config_file="data/Floor_Mappings.yaml", 
                              hospital_data_fields=hospital_data_fields)

    print(processor.hospital_data['temperature_orders'].head(10))


if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")




        

    
