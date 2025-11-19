import argparse
from datetime import datetime
import re
import traceback

import numpy as np
import pandas as pd
import yaml
from typing import Optional

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
        self.blood_pressure_orders_df = self._annotate_blood_pressure_orders_with_room()
        self.heart_rate_orders_df = self._annotate_heart_rate_orders_with_room()
        self.respiratory_rate_orders_df = self._annotate_respiratory_rate_orders_with_room()
        self.temperature_orders_df = self._annotate_temperature_orders_with_room()
        self.oxygen_saturation_orders_df = self._annotate_oxygen_saturation_orders_with_room()
        self._save_processed_data()

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

        scheduled_df = self._annotate_events_with_room(
            events_df=self.hospital_data['medications_orders'],
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

        administered_df.drop(columns=["scheduled_start",
                                  "scheduled_end",
                                  "administered_start", 
                                  "administered_end", 
                                  "Patient ID", 
                                  "Patient Encounter CSN", 
                                  "Race",
                                  "Age at Admission",
                                  "Order Med ID"], inplace=True)
        
        return administered_df
    def parse_frequency_to_timedelta(self, s: str) -> Optional[pd.Timedelta]:
        """Convert an Order Frequency string to a Timedelta.
        Returns NaT when we intentionally want to 'default later'."""
        if not isinstance(s, str):
            return pd.NaT

        txt = s.strip().lower().replace("zzz", "")  # normalize, drop zzz prefix
        if not txt:
            return pd.NaT

        # Explicit PRN-ish or non-scheduled phrases -> treat as unknown
        if txt in {"as needed", "per unit routine", "missing value"}:
            return pd.NaT

        # Daily / times daily
        if txt == "daily":
            return pd.Timedelta(days=1)

        m = re.match(r"(\d+)\s*times\s*daily", txt)
        if m:
            n = int(m.group(1))
            # 24 hours / n (rounded to the nearest minute)
            minutes = round(24 * 60 / max(n, 1))
            return pd.Timedelta(minutes=minutes)

        # Hourly/minutely variants
        if txt == "every hour" or txt == "hourly":
            return pd.Timedelta(hours=1)

        m = re.match(r"every\s+(\d+)\s*hours?", txt)
        if m:
            return pd.Timedelta(hours=int(m.group(1)))

        m = re.match(r"every\s+(\d+)\s*min(?:ute)?s?", txt)
        if m:
            return pd.Timedelta(minutes=int(m.group(1)))

        # Shift-based
        if "shift" in txt:
            return pd.Timedelta(hours=8)

        # Fallback unknown
        return pd.NaT
    
    def _generate_scheduled_times_from_taken_times(self, 
                                                   taken_time_label: str, 
                                                   order_frequency_label: str, 
                                                   df: pd.DataFrame):
        # Parse & sort
        df[taken_time_label] = pd.to_datetime(df[taken_time_label], errors="coerce")

        grp_cols = [self.hospital_data_fields.visits_patient_id_column, 
                    self.hospital_data_fields.visits_encounter_id_column]
        
        df = df.sort_values(grp_cols + [taken_time_label]).reset_index(drop=True)

        freq_td_raw = df[order_frequency_label].apply(self.parse_frequency_to_timedelta)
        df["freq_timedelta"] = freq_td_raw.fillna(pd.Timedelta(hours=4))

        g = df.groupby(grp_cols, group_keys=False)
        curr_taken = df[taken_time_label]
        prev_taken = g[taken_time_label].shift(1)
        elapsed = curr_taken - prev_taken
        valid_prev = prev_taken.notna()

        # Timedelta thresholds
        td10m = pd.Timedelta(minutes=10)
        td2h  = pd.Timedelta(hours=2)

        # Masks (mutually exclusive by construction)
        m_lt10   = valid_prev & (elapsed < td10m)
        m_10_2   = valid_prev & (elapsed >= td10m) & (elapsed < td2h)
        m_gt2    = valid_prev & (elapsed >= td2h)
        m_noprev = ~valid_prev  # first row in each group

        # Build each branch
        sched_lt10 = curr_taken.dt.round("5min")                      # < 10 min → round to nearest 5 min
        sched_10_2 = curr_taken.dt.round("15min")                     # 10 min–2 h → round to nearest 15 min
        sched_gt2  = prev_taken.dt.round("30min") + df["freq_timedelta"]  # > 2 h → prev rounded to 30 min + freq hours
        sched_seed = curr_taken.dt.round("30min")                     # no previous → seed at nearest 30 min

        order_lt10 = curr_taken.dt.round("5min") - pd.Timedelta(minutes=5)
        order_10_2 = curr_taken.dt.round("15min") - pd.Timedelta(minutes=30)
        order_gt2  = prev_taken.dt.round("30min") + df["freq_timedelta"] - pd.Timedelta(minutes=30)
        order_noprev = curr_taken.dt.round("30min") - pd.Timedelta(minutes=30)

        # Combine
        scheduled = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        scheduled.loc[m_lt10]   = sched_lt10[m_lt10]
        scheduled.loc[m_10_2]   = sched_10_2[m_10_2]
        scheduled.loc[m_gt2]    = sched_gt2[m_gt2]
        scheduled.loc[m_noprev] = sched_seed[m_noprev]

        ordered = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        ordered.loc[m_lt10]     = order_lt10[m_lt10]
        ordered.loc[m_10_2]     = order_10_2[m_10_2]
        ordered.loc[m_gt2]      = order_gt2[m_gt2]
        ordered.loc[m_noprev]   = order_noprev[m_noprev]

        df["Scheduled DTTM"] = scheduled
        df["Ordered DTTM"] = ordered

        df.rename(columns={taken_time_label: "Administered DTTM"}, inplace=True)

        return df.drop(columns=["freq_timedelta"])
    
    def _annotate_orders_with_room(self, df: pd.DataFrame) -> pd.DataFrame:

        scheduled_df = self._annotate_events_with_room(
            events_df=df,
            ts_col="Scheduled DTTM",
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

        administered_df.drop(columns=["scheduled_start",
                                  "scheduled_end",
                                  "administered_start", 
                                  "administered_end", 
                                  "Patient Encounter CSN"], inplace=True)
        return administered_df
    
    def _annotate_blood_pressure_orders_with_room(self) -> pd.DataFrame:

        bp_df = self._generate_scheduled_times_from_taken_times(
            taken_time_label="Blood Pressure Taken DTTM",
            order_frequency_label="Order Frequency",
            df=self.hospital_data['blood_pressure_orders']
        )

        bd_annotated_df = self._annotate_orders_with_room(bp_df)
        return bd_annotated_df
    
    def _annotate_heart_rate_orders_with_room(self) -> pd.DataFrame:
        
        hr_df = self._generate_scheduled_times_from_taken_times(
            taken_time_label="Heart Rate Taken DTTM",
            order_frequency_label="Order Frequency",
            df=self.hospital_data['heart_rate_orders']
        )

        hr_annotated_df = self._annotate_orders_with_room(hr_df)
        return hr_annotated_df
    
    def _annotate_respiratory_rate_orders_with_room(self) -> pd.DataFrame:
        
        rr_df = self._generate_scheduled_times_from_taken_times(
            taken_time_label="Respiration DTTM",
            order_frequency_label="Order Frequency",
            df=self.hospital_data['respiratory_rate_orders']
        )

        rr_annotated_df = self._annotate_orders_with_room(rr_df)
        return rr_annotated_df
    
    def _annotate_oxygen_saturation_orders_with_room(self) -> pd.DataFrame:
        
        spo2_df = self._generate_scheduled_times_from_taken_times(
            taken_time_label="SP02 Taken DTTM",
            order_frequency_label="Order Frequency",
            df=self.hospital_data['oxygen_saturation_orders']
        )

        spo2_annotated_df = self._annotate_orders_with_room(spo2_df)
        return spo2_annotated_df
    
    def _annotate_temperature_orders_with_room(self) -> pd.DataFrame:
        
        temp_df = self._generate_scheduled_times_from_taken_times(
            taken_time_label="Temp Taken DTTM",
            order_frequency_label="Order Frequency",
            df=self.hospital_data['temperature_orders']
        )

        temp_annotated_df = self._annotate_orders_with_room(temp_df)
        return temp_annotated_df
    
    def _save_processed_data(self):
        self.medication_orders_df.to_csv("data/processed/medication_orders_annotated.csv", index=False)
        self.blood_pressure_orders_df.to_csv("data/processed/blood_pressure_orders_annotated.csv", index=False)
        self.heart_rate_orders_df.to_csv("data/processed/heart_rate_orders_annotated.csv", index=False)
        self.respiratory_rate_orders_df.to_csv("data/processed/respiratory_rate_orders_annotated.csv", index=False)
        self.temperature_orders_df.to_csv("data/processed/temperature_orders_annotated.csv", index=False)
        self.oxygen_saturation_orders_df.to_csv("data/processed/oxygen_saturation_orders_annotated.csv", index=False)
        self.stays_df.to_csv("data/processed/patient_room_stays.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Process hospital data to extract patient room stays.")
    parser.add_argument("--visits_data_file", type=str, default="data/raw/Visit_Data.csv", help="Path to the visits data CSV file.")
    parser.add_argument("--medications_orders_file", type=str, default="data/raw/Medications_Data.csv", help="Path to the medications orders CSV file.")
    parser.add_argument("--blood_pressure_orders_file", type=str, default="data/raw/Blood_Pressure_Data.csv", help="Path to the blood pressure orders CSV file.")
    parser.add_argument("--heart_rate_orders_file", type=str, default="data/raw/Heart_Rate_Data.csv", help="Path to the heart rate orders CSV file.")
    parser.add_argument("--respiratory_rate_orders_file", type=str, default="data/raw/Respiration_Data.csv", help="Path to the respiratory rate orders CSV file.")
    parser.add_argument("--temperature_orders_file", type=str, default="data/raw/Temperature_Data.csv", help="Path to the temperature orders CSV file.")
    parser.add_argument("--oxygen_saturation_orders_file", type=str, default="data/raw/SP02_Data.csv", help="Path to the oxygen saturation orders CSV file.")
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
                              hospital_config_file="data/map/Floor_Mappings.yaml", 
                              hospital_data_fields=hospital_data_fields)


if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")




        

    
