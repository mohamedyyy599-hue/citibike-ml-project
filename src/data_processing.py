import pandas as pd
import os, zipfile
from pathlib import Path

def aggregate(zip_files: list[Path],  output_path: str):
    df_departures = []
    df_arrivals = []
    
    for z in zip_files:
        with zipfile.ZipFile(z) as zf:
            csv_files = [
                f
                for f in zf.namelist()
                if f.endswith(".csv") and not f.startswith("__")
            ]
            for c in csv_files:
                with zf.open(c) as f:
                    for chunk in pd.read_csv(f, chunksize=500_000, usecols=['started_at', 'ended_at', 'start_station_id', 'end_station_id'], dtype={'start_station_id': str, 'end_station_id': str}):
                        chunk['started_at'] = pd.to_datetime(chunk['started_at'],  format="mixed", errors="coerce").dt.floor('h')
                        chunk['ended_at'] = pd.to_datetime(chunk['ended_at'],  format="mixed", errors="coerce").dt.floor('h')
                        grouped_departures = (
                             chunk.groupby(["start_station_id","started_at"])
                             .size()
                             .reset_index(name="departures") 
                             )
                        grouped_arrivals = (
                             chunk.groupby(["end_station_id","ended_at"])
                             .size()
                             .reset_index(name="arrivals") 
                             )
                        
                        df_departures.append(grouped_departures)
                        df_arrivals.append(grouped_arrivals)
            

    # 1. Concatenate all chunks and rename columns
    big_df_departures = pd.concat(df_departures, ignore_index=True).rename(columns={
        "start_station_id": "station_id",
        "started_at": "hour"
    })

    big_df_arrivals = pd.concat(df_arrivals, ignore_index=True).rename(columns={
        "end_station_id": "station_id",
        "ended_at": "hour"
    })

    # 2. Collapse the duplicates created by chunking
    big_df_departures = big_df_departures.groupby(['station_id', 'hour'])['departures'].sum().reset_index()
    big_df_arrivals = big_df_arrivals.groupby(['station_id', 'hour'])['arrivals'].sum().reset_index()    
            
    merged_df = pd.merge(big_df_departures, big_df_arrivals, on=['station_id', 'hour'], how='outer')
    merged_df = merged_df.fillna(0)
    os.makedirs(output_path , exist_ok=True)
    save_path = os.path.join(output_path, "hourly_demand.parquet")
        # ... your existing merge and fillna code ...

    # ==========================================
    # DATA VALIDATION TESTS (ASSERTS)
    # ==========================================
    print("Running Data Validation Tests...")

    # 1. Test Schema (Column names)
    expected_cols = ['station_id', 'hour', 'departures', 'arrivals']
    assert list(merged_df.columns) == expected_cols, f"Schema mismatch! Expected {expected_cols}, got {list(merged_df.columns)}"

    # 2. Test Data Types
    assert pd.api.types.is_datetime64_any_dtype(merged_df['hour']), "The 'hour' column is not a datetime object!"
    assert pd.api.types.is_numeric_dtype(merged_df['departures']), "Departures is not numeric!"
    assert pd.api.types.is_numeric_dtype(merged_df['arrivals']), "Arrivals is not numeric!"

    # 3. Test for Missing Values (After fillna)
    total_nans = merged_df.isna().sum().sum()
    assert total_nans == 0, f"Data contains {total_nans} missing values! fillna() failed."

    # 4. Test Uniqueness (Grain)
    # A station should only have ONE row per hour.
    duplicates = merged_df.duplicated(subset=['station_id', 'hour']).sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows for the same station and hour!"

    # 5. Test Value Constraints (No negative trips)
    assert (merged_df['departures'] >= 0).all(), "Found negative departure counts!"
    assert (merged_df['arrivals'] >= 0).all(), "Found negative arrival counts!"

    # 6. Test Time Boundaries (Optional but good)
    # Ensure the hours are roughly within the expected month (e.g., Jan 2025)
    min_date = merged_df['hour'].min()
    max_date = merged_df['hour'].max()
    print(f"Data time range: {min_date} to {max_date}")
    
    # If testing 202501 file specifically, it shouldn't have dates from 2024.
    # (You can comment this out when running the full 21 months)
    # assert str(min_date).startswith('2025-01'), "Data contains dates outside the expected month!"

    # 7. Sanity Check on Totals (Optional)
    # In a perfectly closed system, total departures should equal total arrivals.
    # However, Citi Bike trips can end in a different month than they started, 
    # or go to a station not in the dataset. So we allow a small margin of error (e.g., 5%).
    total_deps = merged_df['departures'].sum()
    total_arrs = merged_df['arrivals'].sum()
    diff_pct = abs(total_deps - total_arrs) / max(total_deps, 1) # Prevents division by zero
    print(f"Total Departures: {total_deps:,.0f} | Total Arrivals: {total_arrs:,.0f}")
    assert diff_pct < 0.05, f"Total departures and arrivals differ by {diff_pct*100:.2f}% (Expected < 5%)"

    print("All tests passed successfully!")
    # ==========================================

    merged_df.to_parquet(save_path)


                        



all_zips = sorted(Path("data/raw").glob("*.zip"))
output_path = "data/processed_small"

# Process the first 7 files (Jan 2025 to Jul 2025)
aggregate(all_zips[:7], output_path)