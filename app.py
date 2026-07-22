"""
Energy Load Forecasting — Streamlit prototype.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

st.set_page_config(page_title="Energy Load Forecast", layout="wide")
st.title("⚡ Energy Load Forecasting Dashboard")
st.caption("AI-powered next-hour electricity demand prediction | 1M1B Applied AI for Climate Action")


@st.cache_data
def load_predictions():
    return pd.read_csv(
        os.path.join(DATA_DIR, "test_predictions.csv"), parse_dates=["timestamp"]
    )


@st.cache_resource
def load_model():
    model_path = os.path.join(DATA_DIR, "model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


df = load_predictions()
model = load_model()

# --- Metrics ---
mae = (df["load_mw"] - df["predicted_load_mw"]).abs().mean()
mape = ((df["load_mw"] - df["predicted_load_mw"]).abs() / df["load_mw"]).mean() * 100
from sklearn.metrics import r2_score
r2 = r2_score(df["load_mw"], df["predicted_load_mw"])
peak_actual = df["load_mw"].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.0f} MW")
col2.metric("MAPE", f"{mape:.2f}%")
col3.metric("R²", f"{r2:.4f}")
col4.metric("Peak Load (test set)", f"{peak_actual:,.0f} MW")

st.divider()

# --- Forecast Chart ---
st.subheader("Forecast vs Actual Load")
n_hours = st.slider("Hours to display", 24, 24 * 14, 24 * 7)
plot_df = df.tail(n_hours)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["load_mw"],
        name="Actual",
        line=dict(color="#1f77b4"),
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["predicted_load_mw"],
        name="Predicted",
        line=dict(color="#ff7f0e", dash="dash"),
    )
)
fig.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", height=450)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Feature Importance ---
if model is not None:
    st.subheader("Feature Importance")
    FEATURES = [
        "hour", "day_of_week", "is_weekend", "day_of_year", "month", "week_of_year",
        "load_lag_1", "load_lag_24", "load_lag_168", "load_rolling_24"
    ]
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_fi.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

# --- Hourly Load Pattern (B) ---
st.subheader("Average Load by Hour of Day")
df["hour"] = df["timestamp"].dt.hour
df["day_type"] = df["timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

hourly = df.groupby(["hour", "day_type"])["load_mw"].mean().reset_index()

fig_hour = go.Figure()
for day_type, color in [("Weekday", "#1f77b4"), ("Weekend", "#ff7f0e")]:
    subset = hourly[hourly["day_type"] == day_type]
    fig_hour.add_trace(go.Scatter(
        x=subset["hour"],
        y=subset["load_mw"],
        name=day_type,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5),
    ))
fig_hour.update_layout(
    xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=2),
    yaxis_title="Avg Load (MW)",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_hour, use_container_width=True)

st.divider()

# --- Seasonal Trend (C) ---
st.subheader("Monthly Load Distribution")
df["month"] = df["timestamp"].dt.month
df["month_name"] = df["timestamp"].dt.strftime("%b")
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

monthly_avg = (
    df.groupby("month")["load_mw"]
    .agg(["mean", "max", "min"])
    .reset_index()
)
monthly_avg["month_name"] = [MONTH_ORDER[m - 1] for m in monthly_avg["month"]]

fig_month = go.Figure()
fig_month.add_trace(go.Bar(
    x=monthly_avg["month_name"],
    y=monthly_avg["mean"],
    name="Avg Load",
    marker_color="#1f77b4",
))
fig_month.add_trace(go.Scatter(
    x=monthly_avg["month_name"],
    y=monthly_avg["max"],
    name="Peak Load",
    mode="lines+markers",
    line=dict(color="#d62728", dash="dot", width=2),
    marker=dict(size=6),
))
fig_month.update_layout(
    xaxis=dict(title="Month", categoryorder="array", categoryarray=MONTH_ORDER),
    yaxis_title="Load (MW)",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_month, use_container_width=True)

st.divider()

# --- Peak Hour Alert ---
st.subheader("Peak Hour Alert Simulation")
threshold = st.slider(
    "Alert threshold (MW)",
    int(df["predicted_load_mw"].min()),
    int(df["predicted_load_mw"].max()),
    int(df["predicted_load_mw"].quantile(0.9)),
)
alerts = plot_df[plot_df["predicted_load_mw"] > threshold]
if len(alerts) > 0:
    st.warning(
        f"{len(alerts)} predicted peak-load hours exceed {threshold:,} MW "
        f"in this window — demand response recommended."
    )
    st.dataframe(alerts[["timestamp", "predicted_load_mw"]], use_container_width=True)
else:
    st.success("No predicted peaks exceed threshold in this window.")

st.divider()

# --- About ---
with st.expander("ℹ️ Model & Data Info"):
    st.markdown("""
    **Model:** XGBoost Regressor  
    **Dataset:** PJM East Interconnection hourly energy consumption (2002–2018)  
    **Source:** [Kaggle — Rob Mulla](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)  
    **Features:** hour, day-of-week, weekend flag, day-of-year, month, week-of-year, 1h/24h/168h lag, 24h rolling mean  
    **Train/Test split:** 80/20 chronological  
    **Rows trained on:** ~116,000 hourly observations  
    """)