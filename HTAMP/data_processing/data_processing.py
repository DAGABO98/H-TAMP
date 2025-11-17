import argparse

import numpy as np
import pandas as pd

from HTAMP.data_processing.processing_dataclasses import HospitalDataFiles

class DataProcessor:
    def __init__(self, hospital_data_files: HospitalDataFiles):
        self.hospital_data_files = hospital_data_files
        self.hospital_data = self._load_hospital_data()

    def _load_hospital_data(self) -> dict[str, pd.DataFrame]:
        """Load and concatenate hospital data from multiple CSV files."""
        data_frames = {}
        for file_attr in vars(self.hospital_data_files):
            file_path = getattr(self.hospital_data_files, file_attr)
            df = pd.read_csv(file_path)
            data_frames[file_attr] = df

        return data_frames