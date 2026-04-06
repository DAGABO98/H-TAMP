import pandas as pd
import numpy as np

df = df.copy()
df['Ordered DTTM'] = pd.to_datetime(df['Ordered DTTM'])
df = df.sort_values(['Patient ID', 'Ordered DTTM'])

def requests_in_next_hour(times):
    # times is a Series of datetimes for one patient, already sorted
    t = times.to_numpy(dtype='datetime64[ns]')
    
    # For each request time, find the insertion point of time + 1 hour
    upper = np.searchsorted(t, t + np.timedelta64(1, 'h'), side='right')
    
    # Count how many rows lie after the current row and before/equal to +1 hour
    counts = upper - np.arange(len(t)) - 1
    
    return pd.Series(counts, index=times.index)

df['requests_next_hour'] = (
    df.groupby('Patient ID', group_keys=False)['Ordered DTTM']
      .apply(requests_in_next_hour)
)

# Maximum over the whole dataframe
max_value = df['requests_next_hour'].max()

# Maximum per patient
max_per_patient = (
    df.groupby('Patient ID')['requests_next_hour']
      .max()
      .reset_index(name='max_requests_next_hour')
)

print(df[['Patient ID', 'Ordered DTTM', 'requests_next_hour']])
print("Overall max:", max_value)
print(max_per_patient)