"""
Energy Load Forecasting — Streamlit prototype.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

st.set_page_config(page_title="Energy Load Forecast", layout="wide")
st.title("⚡ Energy Load Forecasting Dashboard")
st.caption("AI-powered next-hour electricity demand prediction | 1M1B Applied AI for Climate Action")

@st.cache_data
def load_predictions():
    return pd.read_csv(os.path.join(DATA_DIR, "test_predictions.csv"), parse_dates=["timestamp"])

df = load_predictions()

col1, col2, col3 = st.columns(3)
mae = (df["load_mw"] - df["predicted_load_mw"]).abs().mean()
mape = ((df["load_mw"] - df["predicted_load_mw"]).abs() / df["load_mw"]).mean() * 100
peak_actual = df["load_mw"].max()

col1.metric("MAE", f"{mae:.1f} MW")
col2.metric("MAPE", f"{mape:.2f}%")
col3.metric("Peak Load (test set)", f"{peak_actual:.0f} MW")

st.subheader("Forecast vs Actual Load")
n_hours = st.slider("Hours to display", 24, 24 * 14, 24 * 7)
plot_df = df.tail(n_hours)

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["load_mw"], name="Actual", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["predicted_load_mw"], name="Predicted", line=dict(color="#ff7f0e", dash="dash")))
fig.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", height=450)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Peak Hour Alert Simulation")
threshold = st.slider("Alert threshold (MW)", int(df["predicted_load_mw"].min()), int(df["predicted_load_mw"].max()), int(df["predicted_load_mw"].quantile(0.9)))
alerts = plot_df[plot_df["predicted_load_mw"] > threshold]
if len(alerts) > 0:
    st.warning(f"{len(alerts)} predicted peak-load hours exceed {threshold} MW in this window — demand response recommended.")
    st.dataframe(alerts[["timestamp", "predicted_load_mw"]], use_container_width=True)
else:
    st.success("No predicted peaks exceed threshold in this window.")

st.divider()
st.caption("Model: XGBoost Regressor | Features: hour, day-of-week, temperature, lag features | Dataset: synthetic hourly load (replace with utility data for production)")
