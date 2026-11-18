import pandas as pd
import json
from pathlib import Path
import numpy as np
def load_data():
    file_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "station_information.json"
    parquet_path = Path(__file__).resolve().parent.parent / "data" / "processed_small" / "hourly_demand.parquet"
    weather_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "weather_nyc_2025_2026.csv"

    df_bikes = pd.read_parquet(parquet_path)
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Extract the nested list of stations
    raw_stations = pd.json_normalize(raw_data["data"]["stations"])
    
    # --- THE REAL FIX ---
    # 1. Select only the columns we need (avoiding the duplicate station_id UUID)
    df_stations = raw_stations[['short_name', 'capacity']]
    
    # 2. Rename short_name to station_id
    df_stations = df_stations.rename(columns={"short_name": "station_id"})
    
    # 3. Clean and deduplicate
    df_stations = df_stations.dropna(subset=['station_id'])
    df_stations['station_id'] = df_stations['station_id'].astype(str)   
    df_stations = df_stations.drop_duplicates(subset=['station_id'], keep='first')

    df_weather = pd.read_csv(weather_path, skiprows=3)
    df_weather['hour'] = pd.to_datetime(df_weather['time']).dt.floor('h')
    df_weather = df_weather[['hour', 'temperature_2m (°C)', 'precipitation (mm)', 'windspeed_10m (km/h)']]
    
    return df_bikes, df_stations, df_weather

def create_dense_grid(df_bikes):
   
    df_bikes_unique = df_bikes['station_id'].unique()
    time_range = pd.date_range(start=df_bikes['hour'].min(),end=df_bikes['hour'].max(),freq='h')
    
    
    full_index = pd.MultiIndex.from_product([df_bikes_unique, time_range], names=['station_id', 'hour'])
   
    df_dense = (
        df_bikes.set_index(['station_id', 'hour'])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return df_dense



def add_time_features(df):
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.day_of_week  # Fixed name
    df['month'] = df['hour'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) # Fixed assignment
    
    return df

def add_lag_features(df):
    df = df.sort_values(['station_id', 'hour']).reset_index(drop=True)
    grouped = df.groupby('station_id')
    df['departures_lag_1h'] = grouped['departures'].shift(1)
    df['departures_lag_24h'] = grouped['departures'].shift(24)
    df['departures_lag_168h'] = grouped['departures'].shift(168)
    df['departures_rolling_mean_24h'] = grouped['departures'].transform(lambda x: x.shift(1).rolling(window=24).mean())
    return df

    

def merge_external_data(df, df_stations, df_weather):
    # 1. Merge stations (left join on station_id)
    df = pd.merge(df, df_stations, on='station_id', how='left')
    
    # 2. Merge weather (left join on hour)
    df = pd.merge(df, df_weather, on='hour', how='left')
    
    # 3. Handle missing weather data (fill forward)
    weather_cols = ['temperature_2m (°C)', 'precipitation (mm)', 'windspeed_10m (km/h)']
    df[weather_cols] = df[weather_cols].ffill().bfill()
    
    # Fill missing capacity just in case (using median capacity)
    df['capacity'] = df['capacity'].fillna(df['capacity'].median())
    
    return df

def create_targets(df):
    # 1. Sort and group
    
    df = df.sort_values(['station_id', 'hour']).reset_index(drop=True)
    grouped = df.groupby('station_id')
    
    # 2. Shift backwards to create future targets
    df['target_departures_2h'] = grouped['departures'].shift(-2)
    df['target_departures_6h'] = grouped['departures'].shift(-6)
    
    # 3. Drop rows where target is NaN (the last 6 hours of each station)
    df = df.dropna(subset=['target_departures_2h', 'target_departures_6h'])
    
    return df


def time_based_split(df):
    train_df = df[df['hour'] < '2025-05-01'].copy()
    val_df = df[(df['hour'] >= '2025-05-01') & (df['hour'] < '2025-06-01')].copy()
    test_df = df[df['hour'] >= '2025-06-01'].copy()
    return train_df,val_df,test_df


def validate_features(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    assert df_train['hour'].max() < df_val['hour'].min()
    assert df_val['hour'].max() < df_test['hour'].min()
    assert {'target_departures_2h','target_departures_6h'}.issubset(df_train.columns)
    assert {'target_departures_2h','target_departures_6h'}.issubset(df_val.columns)
    assert {'target_departures_2h','target_departures_6h'}.issubset(df_test.columns)
    assert df_train['departures_lag_1h'].notna().any()
    assert df_val['departures_lag_1h'].notna().any()
    assert df_test['departures_lag_1h'].notna().any()
    assert {'hour_of_day','day_of_week', 'is_weekend'}.issubset(df_train.columns)
    assert {'hour_of_day','day_of_week', 'is_weekend'}.issubset(df_val.columns)
    assert {'hour_of_day','day_of_week', 'is_weekend'}.issubset(df_test.columns)



if __name__ == "__main__":
    df_bikes, df_stations, df_weather = load_data()
    df_dense = create_dense_grid(df_bikes)
    df_time = add_time_features(df_dense)
    df_lags = add_lag_features(df_time)
    df_merged = merge_external_data(df_lags, df_stations, df_weather)
    df_final = create_targets(df_merged)
    
    # 1. Split the data
    train_df, val_df, test_df = time_based_split(df_final)
    
    # 2. Print the shapes to verify the split worked
    print(f"Train shape (Jan-Apr): {train_df.shape}")
    print(f"Val shape (May): {val_df.shape}")
    print(f"Test shape (Jun-Jul): {test_df.shape}")
    
    # 3. Run the validation tests
    validate_features(train_df, val_df, test_df)
    print("All validations passed successfully!")
    
    # 4. Save the datasets for the ML phase!
    train_df.to_parquet("data/processed_small/train.parquet")
    val_df.to_parquet("data/processed_small/val.parquet")
    test_df.to_parquet("data/processed_small/test.parquet")