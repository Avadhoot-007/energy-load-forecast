import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generate_data import add_features


def _sample_df(n=200):
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
        "load_mw": range(n),
    })


def test_add_features_columns_present():
    out = add_features(_sample_df())
    for col in [
        "hour", "day_of_week", "is_weekend", "day_of_year", "month", "week_of_year",
        "load_lag_1", "load_lag_24", "load_lag_168", "load_rolling_24",
    ]:
        assert col in out.columns


def test_add_features_row_drop_driven_by_longest_lag():
    n = 200
    out = add_features(_sample_df(n))
    # load_lag_168 is the longest lookback -> drives how many leading rows drop
    assert len(out) == n - 168


def test_add_features_lag_values_correct():
    out = add_features(_sample_df())
    assert out["load_lag_1"].iloc[0] == out["load_mw"].iloc[0] - 1
    assert out["load_lag_24"].iloc[0] == out["load_mw"].iloc[0] - 24
    assert out["load_lag_168"].iloc[0] == out["load_mw"].iloc[0] - 168


def test_no_nulls_after_features():
    out = add_features(_sample_df())
    assert out.isnull().sum().sum() == 0


def test_add_features_too_short_input_drops_everything():
    # Fewer than 168 rows -> every row has a NaN load_lag_168 -> empty result
    out = add_features(_sample_df(50))
    assert len(out) == 0