import pandas as pd
import json
import os

def load_and_prepare_data(data_dir):
    """Goal 1: Loads the parquet file, cleans the JSON, and merges them."""
    print("Loading hourly panel...")
    df = pd.read_parquet(os.path.join(data_dir, "hourly_panel.parquet"))
    
    print("Loading station information...")
    with open(os.path.join(data_dir, "station_information.json"), "r") as f:
        stations_json = json.load(f)
        
    stations_df = pd.DataFrame(stations_json['data']['stations'])
    
    # Fix the ID mismatch (drop internal ID, use short_name)
    stations_df = stations_df.drop(columns=['station_id'])
    stations_df = stations_df.rename(columns={'short_name': 'station_id'})
    stations_df = stations_df[['station_id', 'name', 'capacity', 'lat', 'lon']]
    
    # Force strings to merge properly
    df['station_id'] = df['station_id'].astype(str)
    stations_df['station_id'] = stations_df['station_id'].astype(str)
    
    # Drop empty columns and merge
    df = df.drop(columns=['name', 'capacity', 'lat', 'lon'], errors='ignore')
    df = df.merge(stations_df, on='station_id', how='left')
    
    print(f"Data loaded. Missing capacity: {df['capacity'].isna().mean() * 100:.2f}%")
    return df, stations_df