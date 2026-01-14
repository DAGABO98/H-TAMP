
from dataclasses import dataclass

from typing import Optional

import pandas as pd

@dataclass
class HospitalDataFiles:
    medications_orders: str
    blood_pressure_orders: str
    heart_rate_orders: str
    respiratory_rate_orders: str
    temperature_orders: str
    oxygen_saturation_orders: str
    visits_data: str


class HospitalDataFields:
    def __init__(self, 
                 visits_time_in_cols: Optional[list[str]] = None,
                 visits_time_out_cols: Optional[list[str]] = None,
                 visits_dept_in_col: Optional[str] = None,
                 visits_dept_out_col: Optional[str] = None,
                 visits_location_in_col: Optional[str] = None,
                 visits_location_out_col: Optional[str] = None,
                 visits_patient_id_col: Optional[str] = None,
                 visits_encounter_id_col: Optional[str] = None):
        if visits_time_in_cols is None:
            visits_time_in_cols = ["IN_TIME",  
                                "HOSPITAL_ADMISSION"]
        self.visits_time_in_columns = visits_time_in_cols

        if visits_time_out_cols is None:
            visits_time_out_cols = ["OUT_TIME", 
                                 "HOSPITAL_DISCHARGE"]
        self.visits_time_out_columns = visits_time_out_cols

        if visits_location_in_col is None:
            visits_location_in_col = "IN_ROOM"
        self.visits_location_in_column = visits_location_in_col

        if visits_location_out_col is None:
            visits_location_out_col = "OUT_ROOM"
        self.visits_location_out_column = visits_location_out_col

        if visits_dept_in_col is None:
            visits_dept_in_col = "IN_DEP"
        self.visits_dept_in_column = visits_dept_in_col

        if visits_dept_out_col is None:
            visits_dept_out_col = "OUT_DEP"
        self.visits_dept_out_column = visits_dept_out_col

        if visits_patient_id_col is None:
            visits_patient_id_col = "MRN"
        self.visits_patient_id_column = visits_patient_id_col

        if visits_encounter_id_col is None:
            visits_encounter_id_col = "Patient Encounter CSN"
        self.visits_encounter_id_column = visits_encounter_id_col

@dataclass
class AnnotatedDataFiles:
    annotated_visits: str
    annotated_admissions_discharges: str
    annotated_medications: str
    annotated_blood_pressure: str
    annotated_heart_rate: str
    annotated_respiratory_rate: str
    annotated_temperature: str
    annotated_oxygen_saturation: str

@dataclass
class PreprocessedDataFrames:
    visits_df: Optional[pd.DataFrame] = None
    admissions_discharges_df: Optional[pd.DataFrame] = None
    medications_df: Optional[pd.DataFrame] = None
    blood_pressure_df: Optional[pd.DataFrame] = None
    heart_rate_df: Optional[pd.DataFrame] = None
    respiratory_rate_df: Optional[pd.DataFrame] = None
    temperature_df: Optional[pd.DataFrame] = None
    oxygen_saturation_df: Optional[pd.DataFrame] = None