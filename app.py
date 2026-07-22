"""
Energy Load Forecasting — Streamlit dashboard (v2).
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import io
import copy
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridSense · Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — dark grid aesthetic ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
.stApp { background-color: #0d1117; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* KPI Cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
}
.kpi-label { font-size: 11px; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 600; color: #58a6ff; }
.kpi-delta { font-size: 12px; color: #3fb950; margin-top: 4px; }

/* Section headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

/* Live ticker */
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1c2128; border: 1px solid #238636;
    border-radius: 20px; padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #3fb950;
}
.live-dot { width: 7px; height: 7px; background: #3fb950; border-radius: 50%; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Alert banner */
.alert-peak {
    background: rgba(248,81,73,0.1); border: 1px solid #f85149;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
    color: #ffa198; font-size: 14px;
}
.alert-normal {
    background: rgba(63,185,80,0.08); border: 1px solid #238636;
    border-radius: 8px; padding: 14px 18px; margin: 12px 0;
    color: #3fb950; font-size: 14px;
}

/* Nav pills */
.nav-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
.nav-pill {
    background: #21262d; border: 1px solid #30363d;
    border-radius: 6px; padding: 6px 14px;
    font-size: 13px; cursor: pointer; color: #e6edf3;
}
.nav-pill.active { background: #1f6feb; border-color: #58a6ff; color: #fff; }

/* Metric overrides */
[data-testid="metric-container"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #58a6ff !important;
}

/* Sliders */
.stSlider > div > div { background: #21262d !important; }

/* Expander */
.streamlit-expanderHeader { background: #161b22 !important; border: 1px solid #21262d !important; }

/* Dataframe */
.dataframe { background: #161b22 !important; }

/* Download button */
.stDownloadButton > button {
    background: #21262d; border: 1px solid #30363d;
    color: #e6edf3; border-radius: 6px;
}
.stDownloadButton > button:hover { border-color: #58a6ff; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter", color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf3")),
    margin=dict(l=0, r=0, t=30, b=0),
)


def plot_layout(**overrides):
    """
    Merge per-chart overrides into the base PLOT_LAYOUT without
    causing 'multiple values for keyword argument' errors.
    Dict-valued keys (xaxis/yaxis/legend/etc) are shallow-merged;
    everything else is replaced outright.
    """
    layout = copy.deepcopy(PLOT_LAYOUT)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(layout.get(k), dict):
            layout[k].update(v)
        else:
            layout[k] = v
    return layout


# ── Carbon intensity estimate (avg US grid ~0.386 kg CO2/kWh) ────────────────
CO2_KG_PER_MWH = 386  # kg CO2 per MWh (EPA eGRID avg)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    path = os.path.join(DATA_DIR, "test_predictions.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_type"] = df["timestamp"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")
    df["month"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.strftime("%b")
    df["error"] = df["predicted_load_mw"] - df["load_mw"]
    df["abs_error"] = df["error"].abs()
    df["co2_actual_kg"] = df["load_mw"] * CO2_KG_PER_MWH
    df["co2_predicted_kg"] = df["predicted_load_mw"] * CO2_KG_PER_MWH
    return df

@st.cache_resource
def load_model():
    path = os.path.join(DATA_DIR, "model.joblib")
    return joblib.load(path) if os.path.exists(path) else None

df = load_predictions()
model = load_model()

# ── Computed metrics ──────────────────────────────────────────────────────────
mae  = mean_absolute_error(df["load_mw"], df["predicted_load_mw"])
mape = mean_absolute_percentage_error(df["load_mw"], df["predicted_load_mw"]) * 100
r2   = r2_score(df["load_mw"], df["predicted_load_mw"])
peak = df["load_mw"].max()

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-family: JetBrains Mono, monospace; font-size: 18px; font-weight: 600; color: #58a6ff;'>⚡ GridSense</div>
        <div style='font-size: 11px; color: #8b949e; margin-top: 2px;'>Energy Load Forecasting</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 What-If Forecast", "🌍 Carbon Tracker", "💰 ROI Calculator", "🔬 Model Diagnostics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;'>Live Simulation</div>", unsafe_allow_html=True)
    live_mode = st.toggle("Auto-refresh (30s)", value=False)
    if live_mode:
        st.markdown('<div class="live-badge"><div class="live-dot"></div>LIVE</div>', unsafe_allow_html=True)
        import time
        time.sleep(30)
        st.rerun()

    st.markdown("---")
    st.markdown("<div style='font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px; color:#e6edf3; margin-top:6px;'>PJM East · 2002–2018<br>{len(df):,} test rows</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:10px; color:#8b949e;'>1M1B Applied AI for Climate Action</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    # Live header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("<h1 style='font-family: JetBrains Mono, monospace; font-size: 24px; font-weight: 600; color: #e6edf3; margin: 0;'>Energy Load Dashboard</h1>", unsafe_allow_html=True)
        st.caption("PJM East Interconnection · XGBoost Forecast · 1M1B Climate Action")
    with col_h2:
        if live_mode:
            st.markdown('<div class="live-badge" style="float:right"><div class="live-dot"></div>LIVE</div>', unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE", f"{mae:.0f} MW", help="Mean Absolute Error")
    k2.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
    k3.metric("R²", f"{r2:.4f}", help="Coefficient of Determination")
    k4.metric("Peak Load", f"{peak:,.0f} MW", help="Max actual load in test set")

    # ── Forecast chart ──
    st.markdown('<div class="section-header">Forecast vs Actual</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        n_hours = st.slider("Hours to display", 24, 24 * 30, 24 * 7, key="fc_slider")
    with col_ctrl2:
        show_ci = st.checkbox("Confidence band", value=True)

    plot_df = df.tail(n_hours).copy()

    # Simulated confidence interval (±1.5 * rolling std of error)
    err_std = df["error"].rolling(168, min_periods=1).std().iloc[-n_hours:].values
    ci_upper = plot_df["predicted_load_mw"].values + 1.5 * err_std
    ci_lower = plot_df["predicted_load_mw"].values - 1.5 * err_std

    # "Now" = last timestamp in plot window
    now_ts = plot_df["timestamp"].iloc[-1]

    fig = go.Figure()
    if show_ci:
        fig.add_trace(go.Scatter(
            x=pd.concat([plot_df["timestamp"], plot_df["timestamp"].iloc[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself", fillcolor="rgba(88,166,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["load_mw"],
        name="Actual", line=dict(color="#58a6ff", width=1.5)))
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["predicted_load_mw"],
        name="Predicted", line=dict(color="#f0883e", width=1.5, dash="dot")))
    fig.add_vline(x=now_ts, line=dict(color="#3fb950", width=1, dash="dash"),
        annotation_text="NOW", annotation_font_color="#3fb950")
    fig.update_layout(**plot_layout(height=400,
        legend=dict(orientation="h", y=1.08, x=0)))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Load (MW)")
    st.plotly_chart(fig, use_container_width=True)

    # Peak alert
    p90 = df["predicted_load_mw"].quantile(0.9)
    peak_count = (plot_df["predicted_load_mw"] > p90).sum()
    if peak_count > 0:
        st.markdown(f'<div class="alert-peak">⚠️ {peak_count} predicted peak-load hours exceed 90th percentile ({p90:,.0f} MW) — demand response recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-normal">✓ No predicted peaks in this window exceed the 90th percentile threshold.</div>', unsafe_allow_html=True)

    # ── Export ──
    csv_buf = io.BytesIO()
    plot_df[["timestamp", "load_mw", "predicted_load_mw", "error", "abs_error"]].to_csv(csv_buf, index=False)
    st.download_button("⬇ Export this window as CSV", csv_buf.getvalue(),
        file_name="gridsense_forecast.csv", mime="text/csv")

    # ── Hourly + Monthly side by side ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Hourly Load Pattern</div>', unsafe_allow_html=True)
        hourly = df.groupby(["hour", "day_type"])["load_mw"].mean().reset_index()
        fig_h = go.Figure()
        for dt, col in [("Weekday", "#58a6ff"), ("Weekend", "#f0883e")]:
            s = hourly[hourly["day_type"] == dt]
            fig_h.add_trace(go.Scatter(x=s["hour"], y=s["load_mw"], name=dt,
                mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=4)))
        fig_h.update_layout(**plot_layout(height=300,
            xaxis=dict(title="Hour", tickmode="linear", dtick=3, gridcolor="#21262d"),
            yaxis=dict(title="Avg MW", gridcolor="#21262d")))
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Monthly Distribution</div>', unsafe_allow_html=True)
        MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = df.groupby("month")["load_mw"].agg(["mean","max"]).reset_index()
        monthly["month_name"] = [MONTH_ORDER[m-1] for m in monthly["month"]]
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=monthly["month_name"], y=monthly["mean"],
            name="Avg", marker_color="#1f6feb", marker_line_width=0))
        fig_m.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["max"],
            name="Peak", mode="lines+markers",
            line=dict(color="#f85149", width=2, dash="dot"), marker=dict(size=5)))
        fig_m.update_layout(**plot_layout(height=300,
            xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER, gridcolor="#21262d"),
            yaxis=dict(title="MW", gridcolor="#21262d")))
        st.plotly_chart(fig_m, use_container_width=True)

    # ── Feature importance ──
    if model is not None:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        FEATURES = ["hour","day_of_week","is_weekend","day_of_year","month","week_of_year",
                    "load_lag_1","load_lag_24","load_lag_168","load_rolling_24"]
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
        fi_df = fi_df.sort_values("Importance")
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0,"#21262d"],[0.5,"#1f6feb"],[1,"#58a6ff"]],
                line=dict(width=0)
            )
        ))
        fig_fi.update_layout(**plot_layout(height=320,
            xaxis=dict(title="Importance", gridcolor="#21262d"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WHAT-IF FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 What-If Forecast":
    st.markdown("<h1 style='font-family: JetBrains Mono, monospace; font-size: 24px; font-weight: 600; color: #e6edf3;'>What-If Forecast</h1>", unsafe_allow_html=True)
    st.caption("Adjust parameters → get an instant load prediction from the trained model.")

    if model is None:
        st.error("Model not found. Run `python src/train.py` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Time Parameters</div>', unsafe_allow_html=True)
            hour        = st.slider("Hour of day", 0, 23, 14)
            day_of_week = st.slider("Day of week (0=Mon, 6=Sun)", 0, 6, 2)
            month       = st.slider("Month", 1, 12, 7)
            day_of_year = st.slider("Day of year", 1, 365, 180)
            week_of_year= (day_of_year // 7) + 1
            is_weekend  = 1 if day_of_week >= 5 else 0

        with col2:
            st.markdown('<div class="section-header">Lag / History Parameters</div>', unsafe_allow_html=True)
            ref_load = int(df["load_mw"].median())
            lag_1   = st.number_input("Load 1h ago (MW)", 20000, 65000, ref_load)
            lag_24  = st.number_input("Load 24h ago (MW)", 20000, 65000, ref_load)
            lag_168 = st.number_input("Load 168h ago (MW)", 20000, 65000, ref_load)
            roll_24 = st.number_input("24h rolling avg (MW)", 20000, 65000, ref_load)

        features = [[hour, day_of_week, is_weekend, day_of_year, month, week_of_year,
                     lag_1, lag_24, lag_168, roll_24]]
        pred = float(model.predict(features)[0])
        co2  = pred * CO2_KG_PER_MWH / 1000  # tonnes

        p90  = float(df["load_mw"].quantile(0.9))
        is_peak = pred > p90

        st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)
        r1, r2_, r3 = st.columns(3)
        r1.metric("Predicted Load", f"{pred:,.0f} MW")
        r2_.metric("CO₂ Estimate", f"{co2:,.1f} t/h")
        r3.metric("Peak Status", "🔴 PEAK" if is_peak else "🟢 Normal")

        if is_peak:
            st.markdown(f'<div class="alert-peak">⚠️ Predicted load ({pred:,.0f} MW) exceeds 90th percentile ({p90:,.0f} MW). Consider demand-response action.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-normal">✓ Load within normal range. Headroom to peak: {p90 - pred:,.0f} MW.</div>', unsafe_allow_html=True)

        # Multi-hour ahead: vary hour +1 to +6
        st.markdown('<div class="section-header">6-Hour Ahead Outlook</div>', unsafe_allow_html=True)
        hours_ahead = list(range(1, 7))
        ahead_preds = []
        for h in hours_ahead:
            f = [[( hour + h) % 24, day_of_week, is_weekend, day_of_year, month,
                   week_of_year, lag_1, lag_24, lag_168, roll_24]]
            ahead_preds.append(float(model.predict(f)[0]))

        fig_ahead = go.Figure()
        fig_ahead.add_trace(go.Bar(
            x=[f"+{h}h" for h in hours_ahead], y=ahead_preds,
            marker=dict(
                color=["#f85149" if p > p90 else "#1f6feb" for p in ahead_preds],
                line=dict(width=0)
            ),
            text=[f"{p:,.0f}" for p in ahead_preds],
            textposition="outside", textfont=dict(color="#e6edf3", size=11)
        ))
        fig_ahead.add_hline(y=p90, line=dict(color="#f85149", dash="dash", width=1),
            annotation_text="90th pctile", annotation_font_color="#f85149")
        fig_ahead.update_layout(**plot_layout(height=320,
            xaxis=dict(title="Hours ahead", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="MW", gridcolor="#21262d")))
        st.plotly_chart(fig_ahead, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CARBON TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Carbon Tracker":
    st.markdown("<h1 style='font-family: JetBrains Mono, monospace; font-size: 24px; font-weight: 600; color: #e6edf3;'>Carbon Intensity Tracker</h1>", unsafe_allow_html=True)
    st.caption(f"Estimated CO₂ at {CO2_KG_PER_MWH} kg/MWh (EPA eGRID average US grid intensity)")

    n_hours_c = st.slider("Hours to display", 24, 24 * 30, 24 * 7, key="co2_slider")
    co2_df = df.tail(n_hours_c).copy()

    total_co2 = co2_df["co2_actual_kg"].sum() / 1e6  # megatonnes
    avg_intensity = co2_df["co2_actual_kg"].mean() / co2_df["load_mw"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total CO₂ (window)", f"{total_co2:.2f} Mt")
    c2.metric("Avg Intensity", f"{avg_intensity:.0f} kg/MWh")
    c3.metric("Peak CO₂ Hour", f"{co2_df['co2_actual_kg'].max()/1000:,.0f} t")

    st.markdown('<div class="section-header">CO₂ Emission Rate</div>', unsafe_allow_html=True)
    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(
        x=co2_df["timestamp"], y=co2_df["co2_actual_kg"] / 1000,
        name="Actual CO₂ (t/h)", fill="tozeroy",
        fillcolor="rgba(248,81,73,0.12)", line=dict(color="#f85149", width=1.5)
    ))
    fig_co2.add_trace(go.Scatter(
        x=co2_df["timestamp"], y=co2_df["co2_predicted_kg"] / 1000,
        name="Predicted CO₂ (t/h)", line=dict(color="#f0883e", width=1.5, dash="dot")
    ))
    fig_co2.update_layout(**plot_layout(height=380,
        yaxis=dict(title="CO₂ (tonnes/hour)", gridcolor="#21262d")))
    st.plotly_chart(fig_co2, use_container_width=True)

    # Monthly CO2
    st.markdown('<div class="section-header">Monthly CO₂ Footprint</div>', unsafe_allow_html=True)
    MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_co2 = df.groupby("month")["co2_actual_kg"].sum().reset_index()
    monthly_co2["month_name"] = [MONTH_ORDER[m-1] for m in monthly_co2["month"]]
    monthly_co2["co2_kt"] = monthly_co2["co2_actual_kg"] / 1e6

    fig_mco2 = go.Figure(go.Bar(
        x=monthly_co2["month_name"], y=monthly_co2["co2_kt"],
        marker=dict(
            color=monthly_co2["co2_kt"],
            colorscale=[[0,"#1f6feb"],[0.5,"#f0883e"],[1,"#f85149"]],
            line=dict(width=0)
        ),
        text=[f"{v:.1f}" for v in monthly_co2["co2_kt"]],
        textposition="outside", textfont=dict(color="#8b949e", size=10)
    ))
    fig_mco2.update_layout(**plot_layout(height=320,
        xaxis=dict(categoryorder="array", categoryarray=MONTH_ORDER),
        yaxis=dict(title="CO₂ (Mt)", gridcolor="#21262d")))
    st.plotly_chart(fig_mco2, use_container_width=True)

    st.markdown("""
    <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; font-size:13px; color:#8b949e; margin-top:8px;'>
    <b style='color:#e6edf3;'>Methodology note</b><br>
    CO₂ estimates use EPA eGRID 2022 average US grid intensity (386 kg CO₂/MWh).
    Real carbon intensity varies by hour and region — winter gas peakers and summer AC load shift the figure.
    This is a conservative estimate for illustration; PJM's actual intensity is ~350–420 kg/MWh depending on season.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROI CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 ROI Calculator":
    st.markdown("<h1 style='font-family: JetBrains Mono, monospace; font-size: 24px; font-weight: 600; color: #e6edf3;'>Demand Response ROI Calculator</h1>", unsafe_allow_html=True)
    st.caption("Estimate savings from curtailing load during predicted peak hours.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Demand Response Parameters</div>', unsafe_allow_html=True)
        curtail_mw   = st.slider("Curtailment per peak hour (MW)", 100, 5000, 500)
        price_mwh    = st.slider("Peak electricity price ($/MWh)", 50, 500, 150)
        dr_hours_pct = st.slider("% of peak hours with DR activated", 10, 100, 60)
        co2_price    = st.slider("Carbon price ($/tonne CO₂)", 0, 100, 25)

    with col2:
        st.markdown('<div class="section-header">Fleet / Utility Scale</div>', unsafe_allow_html=True)
        n_sites   = st.slider("Number of sites / substations", 1, 100, 10)
        months    = st.slider("Analysis period (months)", 1, 12, 12)

    # Compute
    p90 = df["predicted_load_mw"].quantile(0.9)
    peak_hrs_total = int((df["predicted_load_mw"] > p90).sum())
    peak_hrs_year  = peak_hrs_total  # test set ≈ full year proportional
    activated_hrs  = peak_hrs_year * (dr_hours_pct / 100) * (months / 12)

    energy_saved_mwh = curtail_mw * n_sites * activated_hrs
    revenue_usd      = energy_saved_mwh * price_mwh
    co2_saved_t      = energy_saved_mwh * CO2_KG_PER_MWH / 1000
    carbon_value     = co2_saved_t * co2_price
    total_value      = revenue_usd + carbon_value

    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Energy Saved", f"{energy_saved_mwh:,.0f} MWh")
    r2.metric("Revenue", f"${revenue_usd:,.0f}")
    r3.metric("CO₂ Avoided", f"{co2_saved_t:,.0f} t")
    r4.metric("Total Value", f"${total_value:,.0f}")

    # Sensitivity: vary curtail_mw
    st.markdown('<div class="section-header">Revenue vs Curtailment (sensitivity)</div>', unsafe_allow_html=True)
    curtail_range = np.arange(100, 5100, 100)
    rev_range = curtail_range * n_sites * activated_hrs * price_mwh
    co2_range = curtail_range * n_sites * activated_hrs * CO2_KG_PER_MWH / 1000 * co2_price

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=curtail_range, y=rev_range / 1e6,
        name="Energy Revenue ($M)", line=dict(color="#58a6ff", width=2)))
    fig_roi.add_trace(go.Scatter(x=curtail_range, y=(rev_range + co2_range) / 1e6,
        name="Total incl. Carbon ($M)", line=dict(color="#3fb950", width=2)))
    fig_roi.add_vline(x=curtail_mw, line=dict(color="#f0883e", dash="dash", width=1),
        annotation_text="Your setting", annotation_font_color="#f0883e")
    fig_roi.update_layout(**plot_layout(height=320,
        xaxis=dict(title="Curtailment (MW)", gridcolor="#21262d"),
        yaxis=dict(title="Value ($M)", gridcolor="#21262d")))
    st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown(f"""
    <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px; font-size:13px; color:#8b949e; margin-top:8px;'>
    <b style='color:#e6edf3;'>Assumptions</b><br>
    Peak hours defined as predicted load &gt; 90th percentile ({p90:,.0f} MW).
    {peak_hrs_total:,} peak hours identified in test set.
    Revenue = curtailed MWh × peak price. Carbon value = CO₂ avoided × carbon price.
    Does not account for DR program participation costs or rebound load.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Model Diagnostics":
    st.markdown("<h1 style='font-family: JetBrains Mono, monospace; font-size: 24px; font-weight: 600; color: #e6edf3;'>Model Diagnostics</h1>", unsafe_allow_html=True)
    st.caption("Residual analysis, error distribution, and anomaly detection.")

    # ── Residuals over time ──
    st.markdown('<div class="section-header">Residuals Over Time</div>', unsafe_allow_html=True)
    n_diag = st.slider("Hours", 24, 24*30, 24*14, key="diag_slider")
    diag_df = df.tail(n_diag).copy()

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=diag_df["timestamp"], y=diag_df["error"],
        mode="markers", marker=dict(size=3, color=diag_df["abs_error"],
        colorscale=[[0,"#1f6feb"],[0.5,"#f0883e"],[1,"#f85149"]],
        opacity=0.7), name="Error (MW)"))
    fig_res.add_hline(y=0, line=dict(color="#3fb950", width=1))
    fig_res.add_hline(y=mae*2, line=dict(color="#f85149", dash="dash", width=1),
        annotation_text="2×MAE", annotation_font_color="#f85149")
    fig_res.add_hline(y=-mae*2, line=dict(color="#f85149", dash="dash", width=1))
    fig_res.update_layout(**plot_layout(height=320,
        yaxis=dict(title="Prediction Error (MW)", gridcolor="#21262d")))
    st.plotly_chart(fig_res, use_container_width=True)

    # ── Error distribution ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Error Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure(go.Histogram(
            x=df["error"], nbinsx=80,
            marker=dict(color="#1f6feb", line=dict(color="#0d1117", width=0.3)),
            name="Error"
        ))
        fig_hist.add_vline(x=0, line=dict(color="#3fb950", width=1.5))
        fig_hist.add_vline(x=df["error"].mean(), line=dict(color="#f0883e", dash="dash"),
            annotation_text=f"μ={df['error'].mean():.0f}", annotation_font_color="#f0883e")
        fig_hist.update_layout(**plot_layout(height=300,
            xaxis=dict(title="Error (MW)", gridcolor="#21262d"),
            yaxis=dict(title="Count", gridcolor="#21262d")))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Error by Hour of Day</div>', unsafe_allow_html=True)
        hourly_err = df.groupby("hour")["abs_error"].mean().reset_index()
        fig_herr = go.Figure(go.Bar(
            x=hourly_err["hour"], y=hourly_err["abs_error"],
            marker=dict(
                color=hourly_err["abs_error"],
                colorscale=[[0,"#1f6feb"],[1,"#f85149"]],
                line=dict(width=0)
            )
        ))
        fig_herr.update_layout(**plot_layout(height=300,
            xaxis=dict(title="Hour of Day", gridcolor="#21262d"),
            yaxis=dict(title="Avg Abs Error (MW)", gridcolor="#21262d")))
        st.plotly_chart(fig_herr, use_container_width=True)

    # ── Anomaly detection ──
    st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
    thresh_sigma = st.slider("Anomaly threshold (σ)", 1.5, 4.0, 2.5, 0.1)
    err_mean = df["error"].mean()
    err_std  = df["error"].std()
    anomalies = df[df["error"].abs() > err_mean + thresh_sigma * err_std].copy()

    st.markdown(f"**{len(anomalies)} anomalous hours** detected (|error| > {thresh_sigma}σ)")

    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=df["timestamp"], y=df["load_mw"],
        name="Actual", line=dict(color="#58a6ff", width=1), opacity=0.5))
    fig_anom.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["load_mw"],
        mode="markers", marker=dict(color="#f85149", size=6, symbol="x"),
        name="Anomaly"))
    fig_anom.update_layout(**plot_layout(height=320,
        yaxis=dict(title="Load (MW)", gridcolor="#21262d")))
    st.plotly_chart(fig_anom, use_container_width=True)

    if len(anomalies) > 0:
        st.dataframe(
            anomalies[["timestamp","load_mw","predicted_load_mw","error"]].head(20),
            use_container_width=True
        )

    # ── Summary stats ──
    st.markdown('<div class="section-header">Error Summary Stats</div>', unsafe_allow_html=True)
    stats = pd.DataFrame({
        "Metric": ["MAE","MAPE","R²","RMSE","Bias (mean error)","Std of error","P95 abs error"],
        "Value": [
            f"{mae:.2f} MW",
            f"{mape:.3f}%",
            f"{r2:.6f}",
            f"{np.sqrt((df['error']**2).mean()):.2f} MW",
            f"{df['error'].mean():.2f} MW",
            f"{df['error'].std():.2f} MW",
            f"{df['abs_error'].quantile(0.95):.2f} MW",
        ]
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)