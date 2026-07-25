"""
Trains an XGBoost regressor to forecast next-hour electricity load.
Dataset: PJM East hourly energy consumption (2002-2018), ~145k rows.
Run generate_data.py first to produce energy_load.csv.
"""
import os
import pandas as pd
import joblib
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


# Chronological 3-way split. Fixed splits (not train_test_split's
# shuffle=False 2-way) because early stopping needs its own held-out
# eval set — using the test set for early stopping lets the model peek
# at test-set noise via the stopping point, which leaks into the
# reported test metrics even though no test *row* is ever trained on.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 -> test


def train():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    X, y = df[FEATURES], df[TARGET]
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=25,
    )
    # Early stopping watches the VALIDATION set only. The test set stays
    # untouched until final scoring below.
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    r2 = r2_score(y_test, preds)

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows | Test: {len(X_test):,} rows")
    print(f"MAE:  {mae:.2f} MW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2:   {r2:.4f}")
    print(f"Best iteration: {model.best_iteration} / {model.n_estimators}")

    joblib.dump(model, MODEL_PATH)

    results = df.iloc[X_test.index].copy()
    results["predicted_load_mw"] = preds
    results[["timestamp", "load_mw", "predicted_load_mw"]].to_csv(PREDS_PATH, index=False)

    print(f"Model saved -> {MODEL_PATH}")
    print(f"Predictions saved -> {PREDS_PATH}")
    return {"mae": mae, "mape": mape, "r2": r2}


if __name__ == "__main__":
    train()