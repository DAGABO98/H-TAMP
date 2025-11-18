import argparse
from datetime import datetime
import re
import traceback

import numpy as np
import pandas as pd

from HTAMP.data_processing.processing_dataclasses import HospitalDataFields, HospitalDataFiles

class DataProcessor:
    def __init__(self, hospital_data_files: HospitalDataFiles, hospital_data_fields: HospitalDataFields):
        self.hospital_data_files = hospital_data_files
        self.hospital_data_fields = hospital_data_fields
        self.hospital_data = self._load_hospital_data()

    def _load_hospital_data(self) -> dict[str, pd.DataFrame]:
        """Load and concatenate hospital data from multiple CSV files."""
        data_frames = {}
        for file_attr in vars(self.hospital_data_files):
            file_path = getattr(self.hospital_data_files, file_attr)
            df = pd.read_csv(file_path)
            data_frames[file_attr] = df

        return data_frames
    
    def _compose_location_string(self, department, room) -> str:
        department = str(department).strip()
        department = re.sub(r'\s+', '-', department)
        room = str(room).strip()
        room = re.sub(r'\s+', '-', room)
        return f"{department}_{room}"
    
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

        print(result.head(20))

        return result

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

    processor = DataProcessor(hospital_data_files, hospital_data_fields)
    stays = processor._extract_patient_room_stays()

    # get unique locations
    unique_locations = stays['location'].unique()
    sorted_locations = np.sort(unique_locations)
    print(f"Unique Locations: {sorted_locations}")


if __name__ == "__main__":
    pStart = datetime.now()
    try:
        main()
    except Exception as errorMainContext:
        print("Fail End Process: ", errorMainContext)
        traceback.print_exc()
    pEnd = datetime.now()
    print(f"Total Execution Time: {pEnd - pStart}")




        

    
