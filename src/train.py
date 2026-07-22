"""
Trains an XGBoost regressor to forecast next-hour electricity load.
Dataset: PJM East hourly energy consumption (2002–2018), ~145k rows.
Run generate_data.py first to produce energy_load.csv.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "energy_load.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model.joblib")
PREDS_PATH = os.path.join(BASE_DIR, "data", "test_predictions.csv")

FEATURES = [
    "hour", "day_of_week", "is_weekend", "day_of_year", "month", "week_of_year",
    "load_lag_1", "load_lag_24", "load_lag_168", "load_rolling_24"
]
TARGET = "load_mw"


def train():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    r2 = r2_score(y_test, preds)

    print(f"MAE:  {mae:.2f} MW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")

    joblib.dump(model, MODEL_PATH)

    results = df.iloc[X_test.index].copy()
    results["predicted_load_mw"] = preds
    results[["timestamp", "load_mw", "predicted_load_mw"]].to_csv(PREDS_PATH, index=False)

    print(f"Model saved → {MODEL_PATH}")
    print(f"Predictions saved → {PREDS_PATH}")
    return {"mae": mae, "mape": mape, "r2": r2}


if __name__ == "__main__":
    train()