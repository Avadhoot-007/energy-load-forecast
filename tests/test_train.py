import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import add_lag_features

def test_add_lag_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=30, freq="h"),
        "load_mw": range(30),
    })
    out = add_lag_features(df)
    assert "load_lag_1" in out.columns
    assert "load_lag_24" in out.columns
    assert len(out) == 30 - 24  # rows dropped due to lag_24 NaNs
    assert out["load_lag_1"].iloc[0] == out["load_mw"].iloc[0] - 1

def test_no_nulls_after_lag():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=50, freq="h"),
        "load_mw": range(50),
    })
    out = add_lag_features(df)
    assert out.isnull().sum().sum() == 0
