import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def load_data():
    train_path = Path(__file__).parent.parent / "data" / "processed_small" /  "train.parquet"
    test_path = Path(__file__).parent.parent / "data" / "processed_small" /  "test.parquet"
    val_path = Path(__file__).parent.parent / "data" / "processed_small" / "val.parquet"

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)
    val_df = pd.read_parquet(val_path)
    return train_df, val_df, test_df
def prepare_and_validate_data(train_df, val_df, test_df):
    target = 'target_departures_2h'

    features = [
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "capacity",
        "temperature_2m (°C)",
        "precipitation (mm)",
        "windspeed_10m (km/h)",
        "departures_lag_1h",
        "departures_lag_24h",
        "departures_lag_168h",
        "departures_rolling_mean_24h",
    ]

    

    X_train = train_df[features]
    X_test = test_df[features]
    X_val = val_df[features]

    y_train = train_df[target]
    y_test = test_df[target]
    y_val = val_df[target] 

    print(f"X_train: {X_train.shape}\n")
    print(f"X_test: {X_test.shape}\n")
    print(f"X_val: {X_val.shape}\n")


    print(f"y_train: {y_train.shape}\n")
    print(f"y_test: {y_test.shape}\n")
    print(f"y_val: {y_val.shape}\n")



    # 1. Check Target Leakage
    assert "target_departures_2h" not in features
    assert "target_departures_6h" not in features

    # 2. Check Current-State Leakage
    assert "departures" not in features
    assert "arrivals" not in features

    # 3. Check Identifier Leakage
    assert "station_id" not in features
    assert "hour" not in features



    for col in X_train.columns:
        assert  pd.api.types.is_numeric_dtype(X_train[col]), (
            f"{col} is not a numeric value"
        )

    # If the timestamp is a column named "hour"
    assert train_df["hour"].max() < val_df["hour"].min(), "Data leakage: train overlaps with validation!"
    assert val_df["hour"].max() < test_df["hour"].min(), "Data leakage: validation overlaps with test!"
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_baseline_model(X_train, y_train, X_val, y_val):
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_true=y_val, y_pred=y_pred)
    rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
    print(f"mse: {mae}")
    print(f"rmse: {rmse}")
    return model








if __name__ == "__main__":
    # 1. Load
    train_df, val_df, test_df = load_data()
    
    # 2. Prepare & Validate
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_and_validate_data(train_df, val_df, test_df)
    
    # 3. Train Baseline
    print("Training Baseline Model (Decision Tree)...")
    baseline_model = train_baseline_model(X_train, y_train, X_val, y_val)
    
    