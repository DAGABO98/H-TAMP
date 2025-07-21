import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class TaskVisualizer:
    def __init__(self, visit_data_path="data/Visit_Data.csv", medications_data_path="data/Medications_Data.csv"):
        """
        Initializes the TaskVisualizer with paths to visit and medications data.
        """
        self.visit_data = pd.read_csv(visit_data_path)
        self.medications_data = pd.read_csv(medications_data_path)

        self.preprocess_data()
    
    def preprocess_data(self):
        """
        Preprocesses the visit and medications data.
        This method can be extended to include specific preprocessing steps.
        """
        # Example preprocessing: Convert date columns to datetime
        self.visit_data = self.visit_data.drop_duplicates()
        self.medications_data = self.medications_data.drop_duplicates()

        self.visit_data['IN_TIME'] = pd.to_datetime(self.visit_data['IN_TIME'])
        self.visit_data['OUT_TIME'] = pd.to_datetime(self.visit_data['OUT_TIME'])
        self.medications_data['Medication Order DTTM'] = pd.to_datetime(self.medications_data['Medication Order DTTM'])

    def plot_admissions_frequency(self):
        """
        Plots a heatmap of admissions frequency
        for each hour of the day and each week of the year.
        """
        # Filter for admissions
        admissions = self.visit_data[self.visit_data['EVENT_TYPE'] == 'TRANSFER IN'].copy()

        # Extract week and hour
        admissions['year'] = admissions['IN_TIME'].dt.year
        admissions['week'] = admissions['IN_TIME'].dt.isocalendar().week
        admissions['hour'] = admissions['IN_TIME'].dt.hour

        # Drop rows with missing IN_TIME
        admissions = admissions.dropna(subset=['IN_TIME'])

        # Combine year and week for a unique index
        admissions['year_week'] = admissions['year'].astype(str) + '-' + admissions['week'].astype(str).str.zfill(2)

        # Group by year_week and hour
        freq = admissions.groupby(['year_week', 'hour']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_week', columns='hour', values='count').fillna(0)

        # Plot
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Admissions Frequency Heatmap (by Hour and Year-Week)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Year-Week')
        plt.tight_layout()
        plt.savefig('results/admissions_heatmap.png')
        plt.close()

    def plot_discharges_frequency(self):
        """
        Plots the frequency of patient discharges.
        """
        pass

    def plot_medications_order_frequency(self):
        """
        Plots the frequency of medication orders.
        """
        pass

def main():
    parser = argparse.ArgumentParser(description="Task Visualizer")
    parser.add_argument('--visit_data', type=str, default="data/Visit_Data.csv", help="Path to Visit Data CSV")
    parser.add_argument('--medications_data', type=str, default="data/Medications_Data.csv", help="Path to Medications Data CSV")
    args = parser.parse_args()

    visualizer = TaskVisualizer(visit_data_path=args.visit_data, medications_data_path=args.medications_data)
    visualizer.plot_admissions_frequency()

if __name__ == "__main__":
    main()