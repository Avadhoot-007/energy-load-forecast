"""
Generates realistic hourly electricity load data for prototyping.
Replace with real dataset (e.g. Kaggle 'PJM Hourly Energy Consumption')
before final submission if possible — note in slide 2 (Insights) either way.
"""
import numpy as np
import pandas as pd

def generate_load_data(start="2023-01-01", periods=24*365*2, freq="h", seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq)

    hour = idx.hour
    dow = idx.dayofweek
    doy = idx.dayofyear

    # Base load
    base = 3000

    # Daily pattern: morning + evening peaks
    daily = 500 * np.sin((hour - 6) * np.pi / 12) + 700 * np.exp(-((hour - 19) ** 2) / 8)

    # Weekly pattern: lower on weekends
    weekly = np.where(dow >= 5, -300, 0)

    # Seasonal pattern: summer AC load + winter heating
    seasonal = 400 * np.sin((doy - 172) * 2 * np.pi / 365) ** 2 * np.where(
        (doy > 120) & (doy < 260), 1, 0.3
    )

    noise = rng.normal(0, 100, size=periods)

    load = base + daily + weekly + seasonal + noise
    temp = 25 + 10 * np.sin((doy - 172) * 2 * np.pi / 365) + rng.normal(0, 2, periods)

    df = pd.DataFrame({
        "timestamp": idx,
        "load_mw": np.round(load, 2),
        "temperature_c": np.round(temp, 1),
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": (dow >= 5).astype(int),
        "day_of_year": doy,
    })
    return df

if __name__ == "__main__":
    df = generate_load_data()
    df.to_csv("/home/claude/energy-load-forecast/data/energy_load.csv", index=False)
    print(df.head())
    print(f"Rows: {len(df)}")
