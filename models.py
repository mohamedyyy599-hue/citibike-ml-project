import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, classification_report
import numpy as np

def train_task1_regression(train_df, val_df, test_df, features, target):
    """Trains the demand forecasting model."""
    print("Training Task 1 Regression Model...")
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1)
    
    model.fit(
        train_df[features], train_df[target],
        eval_set=[(val_df[features], val_df[target])],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    preds = model.predict(test_df[features])
    mae = mean_absolute_error(test_df[target], preds)
    rmse = np.sqrt(mean_squared_error(test_df[target], preds))
    
    print(f"Task 1 MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return model

def train_task2_classification(train_df, val_df, test_df, features, target):
    """Trains the risk classification model using Class Weights."""
    print("Training Task 2 Classification Model...")
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    model.fit(
        train_df[features], train_df[target],
        eval_set=[(val_df[features], val_df[target])],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # Get probabilities for threshold tuning
    preds_proba = model.predict_proba(test_df[features])[:, 1]
    return model, preds_proba