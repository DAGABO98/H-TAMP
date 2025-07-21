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

    def plot_hourly_admissions_frequency_per_week(self):
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
        plt.savefig('results/hourly_admissions_heatmap_per_week.png')
        plt.close()
    
    def plot_hourly_discharge_frequency_per_week(self):
        """
        Plots a heatmap of discharge frequency
        for each hour of the day and each week of the year.
        """
        # Filter for discharges
        discharges = self.visit_data[self.visit_data['EVENT_TYPE'] == 'DISCHARGE'].copy()

        # Extract week and hour
        discharges['year'] = discharges['OUT_TIME'].dt.year
        discharges['week'] = discharges['OUT_TIME'].dt.isocalendar().week
        discharges['hour'] = discharges['OUT_TIME'].dt.hour

        # Combine year and week for a unique index
        discharges['year_week'] = discharges['year'].astype(str) + '-' + discharges['week'].astype(str).str.zfill(2)

        # Group by year_week and hour
        freq = discharges.groupby(['year_week', 'hour']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_week', columns='hour', values='count').fillna(0)

        # Plot
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Discharges Frequency Heatmap (by Hour and Year-Week)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Year-Week')
        plt.tight_layout()
        plt.savefig('results/hourly_discharges_heatmap_per_week.png')
        plt.close()
    
    def plot_hourly_medications_frequency_per_week(self):
        """
        Plots a heatmap of medication orders frequency
        for each hour of the day and each week of the year.
        """
        # Extract week and hour from medication order datetime
        self.medications_data['year'] = self.medications_data['Medication Order DTTM'].dt.year
        self.medications_data['week'] = self.medications_data['Medication Order DTTM'].dt.isocalendar().week
        self.medications_data['hour'] = self.medications_data['Medication Order DTTM'].dt.hour

        # Combine year and week for a unique index
        self.medications_data['year_week'] = self.medications_data['year'].astype(str) + '-' + self.medications_data['week'].astype(str).str.zfill(2)

        # Group by year_week and hour
        freq = self.medications_data.groupby(['year_week', 'hour']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_week', columns='hour', values='count').fillna(0)

        # Plot
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Medications Orders Frequency Heatmap (by Hour and Year-Week)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Year-Week')
        plt.tight_layout()
        plt.savefig('results/hourly_medications_heatmap_per_week.png')
        plt.close()
    
    def plot_hourly_admissions_frequency_per_month(self):
        """
        Plots a heatmap of admissions frequency
        for each hour of the day and each month of the year.
        """
        admissions = self.visit_data[self.visit_data['EVENT_TYPE'] == 'TRANSFER IN'].copy()
        admissions = admissions.dropna(subset=['IN_TIME'])

        admissions['year'] = admissions['IN_TIME'].dt.year
        admissions['month'] = admissions['IN_TIME'].dt.month
        admissions['hour'] = admissions['IN_TIME'].dt.hour

        # Combine year and month for a unique index
        admissions['year_month'] = admissions['year'].astype(str) + '-' + admissions['month'].astype(str).str.zfill(2)

        # Group by year_month and hour
        freq = admissions.groupby(['year_month', 'hour']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_month', columns='hour', values='count').fillna(0)

        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Admissions Frequency Heatmap (by Hour and Year-Month)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Year-Month')
        plt.tight_layout()
        plt.savefig('results/hourly_admissions_heatmap_per_month.png')
        plt.close()
    
    def plot_hourly_discharge_frequency_per_month(self):
        """
        Plots a heatmap of discharge frequency
        for each hour of the day and each month of the year.
        """
        discharges = self.visit_data[self.visit_data['EVENT_TYPE'] == 'DISCHARGE'].copy()
        discharges = discharges.dropna(subset=['OUT_TIME'])

        discharges['year'] = discharges['OUT_TIME'].dt.year
        discharges['month'] = discharges['OUT_TIME'].dt.month
        discharges['hour'] = discharges['OUT_TIME'].dt.hour

        # Combine year and month for a unique index
        discharges['year_month'] = discharges['year'].astype(str) + '-' + discharges['month'].astype(str).str.zfill(2)

        # Group by year_month and hour
        freq = discharges.groupby(['year_month', 'hour']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_month', columns='hour', values='count').fillna(0)

        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Discharges Frequency Heatmap (by Hour and Year-Month)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Year-Month')
        plt.tight_layout()
        plt.savefig('results/hourly_discharges_heatmap_per_month.png')
        plt.close()
    
    def plot_hourly_medications_frequency_per_month(self):
        """
        Plots a heatmap of medication orders frequency
        for each hour of the day and each month of the year.
        """
        self.medications_data['year'] = self.medications_data['Medication Order DTTM'].dt.year
        self.medications_data['month'] = self.medications_data['Medication Order DTTM'].dt.month
        self.medications_data['hour'] = self.medications_data['Medication Order DTTM'].dt.hour

        # Combine year and month for a unique index
        self.medications_data['year_month'] = self.medications_data['year'].astype(str) + '-' + self.medications_data['month'].astype(str).str.zfill(2)

        # Group by year_month and hour
        freq = self.medications_data.groupby(['year_month', 'hour']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_month', columns='hour', values='count').fillna(0)

        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Medications Orders Frequency Heatmap (by Hour and Year-Month)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Year-Month')
        plt.tight_layout()
        plt.savefig('results/hourly_medications_heatmap_per_month.png')
        plt.close()
    
    def plot_daily_admissions_frequency_per_week(self):
        """
        Plots a heatmap of admissions frequency
        for each day of the week and each week of the year.
        """
        admissions = self.visit_data[self.visit_data['EVENT_TYPE'] == 'TRANSFER IN'].copy()
        admissions = admissions.dropna(subset=['IN_TIME'])

        admissions['year'] = admissions['IN_TIME'].dt.year
        admissions['week'] = admissions['IN_TIME'].dt.isocalendar().week
        admissions['dayofweek'] = admissions['IN_TIME'].dt.dayofweek  # Monday=0, Sunday=6

        # Combine year and week for a unique index
        admissions['year_week'] = admissions['year'].astype(str) + '-' + admissions['week'].astype(str).str.zfill(2)

        # Group by year_week and dayofweek
        freq = admissions.groupby(['year_week', 'dayofweek']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_week', columns='dayofweek', values='count').fillna(0)

        # Optional: Set day names as column labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data.columns = day_names

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Admissions Frequency Heatmap (by Day of Week and Year-Week)')
        plt.xlabel('Day of Week')
        plt.ylabel('Year-Week')
        plt.tight_layout()
        plt.savefig('results/daily_admissions_heatmap_per_week.png')
        plt.close()
    
    def plot_daily_discharge_frequency_per_week(self):
        """
        Plots a heatmap of discharge frequency
        for each day of the week and each week of the year.
        """
        discharges = self.visit_data[self.visit_data['EVENT_TYPE'] == 'DISCHARGE'].copy()
        discharges = discharges.dropna(subset=['OUT_TIME'])

        discharges['year'] = discharges['OUT_TIME'].dt.year
        discharges['week'] = discharges['OUT_TIME'].dt.isocalendar().week
        discharges['dayofweek'] = discharges['OUT_TIME'].dt.dayofweek

        # Combine year and week for a unique index
        discharges['year_week'] = discharges['year'].astype(str) + '-' + discharges['week'].astype(str).str.zfill(2)

        # Group by year_week and dayofweek
        freq = discharges.groupby(['year_week', 'dayofweek']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_week', columns='dayofweek', values='count').fillna(0)

        # Optional: Set day names as column labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data.columns = day_names

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Discharges Frequency Heatmap (by Day of Week and Year-Week)')
        plt.xlabel('Day of Week')
        plt.ylabel('Year-Week')
        plt.tight_layout()
        plt.savefig('results/daily_discharges_heatmap_per_week.png')
        plt.close()
    
    def plot_daily_medications_frequency_per_week(self):
        """
        Plots a heatmap of medication orders frequency
        for each day of the week and each week of the year.
        """
        self.medications_data['year'] = self.medications_data['Medication Order DTTM'].dt.year
        self.medications_data['week'] = self.medications_data['Medication Order DTTM'].dt.isocalendar().week
        self.medications_data['dayofweek'] = self.medications_data['Medication Order DTTM'].dt.dayofweek

        # Combine year and week for a unique index
        self.medications_data['year_week'] = self.medications_data['year'].astype(str) + '-' + self.medications_data['week'].astype(str).str.zfill(2)

        # Group by year_week and dayofweek
        freq = self.medications_data.groupby(['year_week', 'dayofweek']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_week', columns='dayofweek', values='count').fillna(0)

        # Optional: Set day names as column labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data.columns = day_names

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Medications Orders Frequency Heatmap (by Day of Week and Year-Week)')
        plt.xlabel('Day of Week')
        plt.ylabel('Year-Week')
        plt.tight_layout()
        plt.savefig('results/daily_medications_heatmap_per_week.png')
        plt.close()
    
    def plot_daily_admissions_frequency_per_month(self):
        """
        Plots a heatmap of admissions frequency
        for each day of the week and each month of the year.
        """
        admissions = self.visit_data[self.visit_data['EVENT_TYPE'] == 'TRANSFER IN'].copy()
        admissions = admissions.dropna(subset=['IN_TIME'])

        admissions['year'] = admissions['IN_TIME'].dt.year
        admissions['month'] = admissions['IN_TIME'].dt.month
        admissions['dayofweek'] = admissions['IN_TIME'].dt.dayofweek  # Monday=0, Sunday=6

        # Combine year and month for a unique index
        admissions['year_month'] = admissions['year'].astype(str) + '-' + admissions['month'].astype(str).str.zfill(2)

        # Group by year_month and dayofweek
        freq = admissions.groupby(['year_month', 'dayofweek']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_month', columns='dayofweek', values='count').fillna(0)

        # Optional: Set day names as column labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data.columns = day_names

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Admissions Frequency Heatmap (by Day of Week and Year-Month)')
        plt.xlabel('Day of Week')
        plt.ylabel('Year-Month')
        plt.tight_layout()
        plt.savefig('results/daily_admissions_heatmap_per_month.png')
        plt.close()
    
    def plot_daily_discharge_frequency_per_month(self):
        """
        Plots a heatmap of discharge frequency
        for each day of the week and each month of the year.
        """
        discharges = self.visit_data[self.visit_data['EVENT_TYPE'] == 'DISCHARGE'].copy()
        discharges = discharges.dropna(subset=['OUT_TIME'])

        discharges['year'] = discharges['OUT_TIME'].dt.year
        discharges['month'] = discharges['OUT_TIME'].dt.month
        discharges['dayofweek'] = discharges['OUT_TIME'].dt.dayofweek

        # Combine year and month for a unique index
        discharges['year_month'] = discharges['year'].astype(str) + '-' + discharges['month'].astype(str).str.zfill(2)

        # Group by year_month and dayofweek
        freq = discharges.groupby(['year_month', 'dayofweek']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_month', columns='dayofweek', values='count').fillna(0)

        # Optional: Set day names as column labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data.columns = day_names

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Discharges Frequency Heatmap (by Day of Week and Year-Month)')
        plt.xlabel('Day of Week')
        plt.ylabel('Year-Month')
        plt.tight_layout()
        plt.savefig('results/daily_discharges_heatmap_per_month.png')
        plt.close()
    
    def plot_daily_medications_frequency_per_month(self):
        """
        Plots a heatmap of medication orders frequency
        for each day of the week and each month of the year.
        """
        self.medications_data['year'] = self.medications_data['Medication Order DTTM'].dt.year
        self.medications_data['month'] = self.medications_data['Medication Order DTTM'].dt.month
        self.medications_data['dayofweek'] = self.medications_data['Medication Order DTTM'].dt.dayofweek

        # Combine year and month for a unique index
        self.medications_data['year_month'] = self.medications_data['year'].astype(str) + '-' + self.medications_data['month'].astype(str).str.zfill(2)

        # Group by year_month and dayofweek
        freq = self.medications_data.groupby(['year_month', 'dayofweek']).size().reset_index(name='count')

        # Pivot to matrix
        heatmap_data = freq.pivot(index='year_month', columns='dayofweek', values='count').fillna(0)

        # Optional: Set day names as column labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        heatmap_data.columns = day_names

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Medications Orders Frequency Heatmap (by Day of Week and Year-Month)')
        plt.xlabel('Day of Week')
        plt.ylabel('Year-Month')
        plt.tight_layout()
        plt.savefig('results/daily_medications_heatmap_per_month.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Task Visualizer")
    parser.add_argument('--visit_data', type=str, default="data/Visit_Data.csv", help="Path to Visit Data CSV")
    parser.add_argument('--medications_data', type=str, default="data/Medications_Data.csv", help="Path to Medications Data CSV")
    args = parser.parse_args()

    visualizer = TaskVisualizer(visit_data_path=args.visit_data, medications_data_path=args.medications_data)

    #hourly results per week
    visualizer.plot_hourly_admissions_frequency_per_week()
    visualizer.plot_hourly_discharge_frequency_per_week()
    visualizer.plot_hourly_medications_frequency_per_week()

    #hourly results per month
    visualizer.plot_hourly_admissions_frequency_per_month()
    visualizer.plot_hourly_discharge_frequency_per_month()
    visualizer.plot_hourly_medications_frequency_per_month()
    
    #daily results per week
    visualizer.plot_daily_admissions_frequency_per_week()
    visualizer.plot_daily_discharge_frequency_per_week()
    visualizer.plot_daily_medications_frequency_per_week()

    #daily results per month
    visualizer.plot_daily_admissions_frequency_per_month()
    visualizer.plot_daily_discharge_frequency_per_month()
    visualizer.plot_daily_medications_frequency_per_month()

if __name__ == "__main__":
    main()