import pandas as pd
import numpy as np

def engineer_features(df, n_stations=300):
    """Samples stations, creates features, lags, and targets."""
    print(f"Sampling {n_stations} stations for memory optimization...")
    np.random.seed(42)
    sample_stations = np.random.choice(df['station_id'].unique(), size=n_stations, replace=False)
    df_sample = df[df['station_id'].isin(sample_stations)].copy()
    
    # --- TIME FEATURES ---
    df_sample['hour'] = pd.to_datetime(df_sample['hour'])
    df_sample['hour_of_day'] = df_sample['hour'].dt.hour
    df_sample['day_of_week'] = df_sample['hour'].dt.day_of_week
    df_sample['month'] = df_sample['hour'].dt.month
    df_sample['is_weekend'] = df_sample['day_of_week'].isin([5, 6]).astype(int)
    df_sample['hour_sin'] = np.sin(2 * np.pi * df_sample['hour_of_day'] / 24)
    df_sample['hour_cos'] = np.cos(2 * np.pi * df_sample['hour_of_day'] / 24)
    
    # --- CHAPTER 5: CROSSING & DOMAIN ---
    df_sample['is_weekend_x_hour'] = df_sample['is_weekend'] * df_sample['hour_of_day']
    
    ts_lat, ts_lon = 40.7580, -73.9855
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371 
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    
    df_sample['dist_from_center_km'] = haversine(df_sample['lat'], df_sample['lon'], ts_lat, ts_lon)
    
    # --- LAGS & TARGETS ---
    df_sample = df_sample.sort_values(by=['station_id', 'hour']).reset_index(drop=True)
    
    # Net Flow (for Task 2)
    df_sample['net_flow_2h'] = df_sample.groupby('station_id')['departures'].transform(
        lambda x: x.rolling(2).sum()
    ) - df_sample.groupby('station_id')['arrivals'].transform(
        lambda x: x.rolling(2).sum()
    )
    
    # Targets
    df_sample['target_departures_2h'] = df_sample.groupby('station_id')['departures'].shift(-2)
    df_sample['target_high_risk_2h'] = df_sample.groupby('station_id')['net_flow_2h'].shift(-2)
    
    # Lags
    df_sample['departures_lag_1h'] = df_sample.groupby('station_id')['departures'].shift(1)
    df_sample['arrivals_lag_1h'] = df_sample.groupby('station_id')['arrivals'].shift(1)
    df_sample['departures_roll_mean_3h'] = df_sample.groupby('station_id')['departures'].transform(lambda x: x.shift(1).rolling(3).mean())
    df_sample['departures_roll_mean_24h'] = df_sample.groupby('station_id')['departures'].transform(lambda x: x.shift(1).rolling(24).mean())
    
    return df_sample.dropna().reset_index(drop=True)

def time_based_split(df):
    """Splits data chronologically and creates the risk label."""
    train_end = '2025-12-31 23:59:59'
    val_end = '2026-04-30 23:59:59'
    
    train_df = df[df['hour'] <= train_end].copy()
    val_df = df[(df['hour'] > train_end) & (df['hour'] <= val_end)].copy()
    test_df = df[df['hour'] > val_end].copy()
    
    # Chapter 4: Programmatic Labeling (Threshold from Train only)
    threshold = train_df.groupby('station_id')['net_flow_2h'].quantile(0.10)
    df['threshold_empty'] = df['station_id'].map(threshold)
    df['is_high_risk_now'] = (df['net_flow_2h'] < df['threshold_empty']).astype(int)
    df['target_high_risk_2h'] = df.groupby('station_id')['is_high_risk_now'].shift(-2)
    
    # Re-split after creating the label
    df = df.dropna(subset=['target_high_risk_2h'])
    train_df = df[df['hour'] <= train_end].copy()
    val_df = df[(df['hour'] > train_end) & (df['hour'] <= val_end)].copy()
    test_df = df[df['hour'] > val_end].copy()
    
    return train_df, val_df, test_df