# Energy Load Forecasting ⚡

AI-powered next-hour electricity demand prediction for smarter grid management.
Built for **1M1B Internship — Green Skills & Applied AI for Climate Action**.

## Problem

Electric grids must match supply to demand in real time. Under-forecasting causes
blackouts; over-forecasting wastes energy and increases emissions. Most utilities,
especially smaller ones, lack accessible forecasting tools.

## Solution

An XGBoost regression model predicts next-hour load using calendar (hour, day of
week, day of year, week of year) and lag/rolling features (1h, 24h, 168h lags,
24h rolling average). A Streamlit dashboard visualizes forecast vs actual load
and flags predicted peak-demand hours for demand-response action.

## Results

| Metric | Value  |
| ------ | ------ |
| MAE    | ~85 MW |
| MAPE   | ~2.7%  |
| R²     | ~0.94  |

## Tech Stack

Python · pandas · XGBoost · scikit-learn · Streamlit · Plotly

## Project Structure

```
energy-load-forecast/
├── data/               # dataset + trained model + predictions (gitignored)
├── notebooks/          # exploration (optional)
├── src/
│   ├── generate_data.py
│   └── train.py
├── tests/
│   └── test_generate_data.py
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt

# 1. Get the dataset: Kaggle "PJM Hourly Energy Consumption" (PJME_hourly.csv)
#    https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
#    Place it at data/PJME_hourly.csv

python src/generate_data.py   # -> data/energy_load.csv
python src/train.py           # -> data/model.joblib, data/test_predictions.csv
streamlit run app.py
```

## Data Note

The model trains on **real PJM East hourly load data** (2002–2018, ~145k rows)
from the Kaggle PJM Hourly Energy Consumption dataset — not synthetic data.
`data/PJME_hourly.csv` is not committed (gitignored, regenerate/download it
yourself); `generate_data.py` turns it into the feature-engineered
`energy_load.csv` that `train.py` consumes. Swapping in a different utility's
data requires matching the `timestamp` / load-value CSV schema — no code
changes to the pipeline itself.

There's currently no weather/temperature feature — forecasts rely on
calendar and lag structure only. Adding temperature is a natural next step
(see below).

## What's Next

- Add a weather/temperature feature (join against a public weather API)
- Multi-region forecasting
- Integrate renewable generation forecast (solar/wind offset)
- Deploy as REST API (AWS Lambda, free tier)
- Real-time SMS/push demand-response alerts
- Proper train/val/test split for hyperparameter tuning without test-set leakage

## Author

Solo project — 1M1B Applied AI for Climate Action internship.
