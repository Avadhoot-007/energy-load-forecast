"""
Trains an XGBoost regressor to forecast next-hour electricity load.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

DATA_PATH = "/home/claude/energy-load-forecast/data/energy_load.csv"
MODEL_PATH = "/home/claude/energy-load-forecast/data/model.joblib"

FEATURES = ["hour", "day_of_week", "is_weekend", "day_of_year", "temperature_c", "load_lag_1", "load_lag_24"]
TARGET = "load_mw"

def add_lag_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["load_lag_1"] = df["load_mw"].shift(1)
    df["load_lag_24"] = df["load_mw"].shift(24)
    df = df.dropna().reset_index(drop=True)
    return df

def train():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = add_lag_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # keep time order
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f} MW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")

    joblib.dump(model, MODEL_PATH)

    # Save test predictions for dashboard
    results = df.iloc[X_test.index].copy()
    results["predicted_load_mw"] = preds
    results[["timestamp", "load_mw", "predicted_load_mw"]].to_csv(
        "/home/claude/energy-load-forecast/data/test_predictions.csv", index=False
    )

    return {"mae": mae, "mape": mape, "r2": r2}

if __name__ == "__main__":
    train()
