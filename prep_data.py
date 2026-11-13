import pandas as pd
import glob
import json
import os
import zipfile

# 1. Define paths
DATA_DIR = os.path.expanduser("~/project_data")
zip_files = glob.glob(os.path.join(DATA_DIR, "*.zip"))

# 2. Load Station Capacity Data
print("Loading station information...")
with open(os.path.join(DATA_DIR, "station_information.json"), "r") as f:
    station_data = json.load(f)

stations_df = pd.DataFrame(station_data['data']['stations'])
stations_df = stations_df[['station_id', 'name', 'capacity', 'lat', 'lon']]
print(f"Loaded capacity data for {len(stations_df)} stations.")

# 3. Load and Aggregate Trip Data (Memory Efficient!)
print(f"Loading and aggregating trip data from {len(zip_files)} files...")
cols_to_keep = ['started_at', 'ended_at', 'start_station_id', 'end_station_id']

# Lists to hold the small, aggregated chunks
departures_list = []
arrivals_list = []

for f in zip_files:
    try:
        with zipfile.ZipFile(f, 'r') as z:
            csv_files = [name for name in z.namelist() if name.endswith('.csv')]
            for csv_name in csv_files:
                with z.open(csv_name) as csv_file:
                    # Force IDs to be strings to fix the DtypeWarning
                    df = pd.read_csv(csv_file, usecols=cols_to_keep, dtype={'start_station_id': str, 'end_station_id': str})
                    
                    df['started_at'] = pd.to_datetime(df['started_at'])
                    df['ended_at'] = pd.to_datetime(df['ended_at'])
                    
                    df['start_hour'] = df['started_at'].dt.floor('h')
                    df['end_hour'] = df['ended_at'].dt.floor('h')
                    
                    # Aggregate IMMEDIATELY to save memory
                    dep = df.groupby(['start_hour', 'start_station_id']).size().reset_index(name='departures')
                    arr = df.groupby(['end_hour', 'end_station_id']).size().reset_index(name='arrivals')
                    
                    departures_list.append(dep)
                    arrivals_list.append(arr)
    except Exception as e:
        print(f"Error reading {f}: {e}")

# 4. Combine the small aggregated chunks
print("Merging aggregated data...")
# Groupby again just in case a month was split into multiple CSVs
departures = pd.concat(departures_list, ignore_index=True).groupby(['start_hour', 'start_station_id']).sum().reset_index()
arrivals = pd.concat(arrivals_list, ignore_index=True).groupby(['end_hour', 'end_station_id']).sum().reset_index()


# Rename columns for merging
departures = departures.rename(columns={'start_hour': 'hour', 'start_station_id': 'station_id'})
arrivals = arrivals.rename(columns={'end_hour': 'hour', 'end_station_id': 'station_id'})

# 5. Create the final Hourly Panel
hourly_panel = pd.merge(departures, arrivals, on=['hour', 'station_id'], how='outer').fillna(0)
# Force both station_id columns to be strings so they match perfectly
hourly_panel['station_id'] = hourly_panel['station_id'].astype(str)
stations_df['station_id'] = stations_df['station_id'].astype(str)

# Now merge
hourly_panel = hourly_panel.merge(stations_df, on='station_id', how='left')

print("\n--- Data Prep Complete ---")
print(hourly_panel.head())
print(f"Final hourly panel shape: {hourly_panel.shape}")

# 6. Save to a new Parquet file
output_path = os.path.join(DATA_DIR, "hourly_panel.parquet")
hourly_panel.to_parquet(output_path, index=False)
print(f"Saved final dataset to {output_path}")