import pandas as pd
import numpy as np

def calculate_priorities(test_df, preds_proba, stations_df, threshold=0.7):
    """Calculates the Task 3 priority score and returns the ranked stations."""
    station_summary = pd.DataFrame()
    
    # 1. Metrics
    station_summary['avg_demand'] = test_df.groupby('station_id')['target_departures_2h'].mean()
    test_df['pred_is_risk'] = (preds_proba >= threshold).astype(int)
    station_summary['risk_frequency'] = test_df.groupby('station_id')['pred_is_risk'].mean()
    
    # Volume
    station_summary['total_volume'] = test_df.groupby('station_id').agg(
        total_deps=('departures', 'sum'),
        total_arrs=('arrivals', 'sum')
    ).sum(axis=1)
    
    # 2. Normalize
    def min_max_scale(series):
        return (series - series.min()) / (series.max() - series.min())
        
    station_summary['norm_demand'] = min_max_scale(station_summary['avg_demand'])
    station_summary['norm_risk'] = min_max_scale(station_summary['risk_frequency'])
    station_summary['norm_volume'] = min_max_scale(station_summary['total_volume'])
    
    # 3. Score
    station_summary['priority_score'] = (
        (station_summary['norm_demand'] * 0.40) + 
        (station_summary['norm_risk'] * 0.40) + 
        (station_summary['norm_volume'] * 0.20)
    )
    
    # 4. Rank and Merge Names
    ranked = station_summary.sort_values(by='priority_score', ascending=False).reset_index()
    
    stations_clean = stations_df[['station_id', 'name', 'capacity']].drop_duplicates()
    stations_clean['station_id'] = stations_clean['station_id'].astype(str)
    ranked['station_id'] = ranked['station_id'].astype(str)
    
    ranked = ranked.merge(stations_clean, on='station_id', how='left')
    return ranked