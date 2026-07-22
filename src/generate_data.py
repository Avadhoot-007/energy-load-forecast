"""
Processes real PJM East hourly energy consumption data.
Source: Kaggle — 'PJM Hourly Energy Consumption' (PJME_hourly.csv)
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

Place PJME_hourly.csv in data/ and run this script to generate energy_load.csv.
"""
import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "PJME_hourly.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "energy_load.csv")


def process_pjm(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["Datetime"])
    df = df.rename(columns={"Datetime": "timestamp", "PJME_MW": "load_mw"})
    df = df.dropna()
    df = df.drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    # Lag features
    df["load_lag_1"] = df["load_mw"].shift(1)
    df["load_lag_24"] = df["load_mw"].shift(24)
    df["load_lag_168"] = df["load_mw"].shift(168)
    df["load_rolling_24"] = df["load_mw"].shift(1).rolling(24).mean()

    df = df.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows → {out_path}")
    return df


if __name__ == "__main__":
    df = process_pjm()
    print(df.head())
    print(f"\nDate range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Load range: {df['load_mw'].min():.0f} – {df['load_mw'].max():.0f} MW")