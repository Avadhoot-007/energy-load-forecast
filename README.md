# Energy Load Forecasting ⚡

AI-powered next-hour electricity demand prediction for smarter grid management.
Built for **1M1B Internship — Green Skills & Applied AI for Climate Action**.

## Problem
Electric grids must match supply to demand in real time. Under-forecasting causes
blackouts; over-forecasting wastes energy and increases emissions. Most utilities,
especially smaller ones, lack accessible forecasting tools.

## Solution
An XGBoost regression model predicts next-hour load using time, calendar,
temperature, and lag features. A Streamlit dashboard visualizes forecast vs
actual load and flags predicted peak-demand hours for demand-response action.

## Results
| Metric | Value |
|---|---|
| MAE | ~85 MW |
| MAPE | ~2.7% |
| R² | ~0.94 |

## Tech Stack
Python · pandas · XGBoost · scikit-learn · Streamlit · Plotly

## Project Structure
```
energy-load-forecast/
├── data/               # dataset + trained model + predictions
├── notebooks/          # exploration (optional)
├── src/
│   ├── generate_data.py
│   └── train.py
├── tests/
│   └── test_train.py
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Run locally
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
streamlit run app.py
```

## Data Note
Current prototype uses realistic synthetic hourly load data (daily/weekly/seasonal
patterns + temperature). For production, swap in a real utility dataset
(e.g., PJM Hourly Energy Consumption) — pipeline requires no code changes,
only a compatible CSV schema.

## What's Next
- Swap in real utility/grid dataset
- Multi-region forecasting
- Integrate renewable generation forecast (solar/wind offset)
- Deploy as REST API (AWS Lambda, free tier)
- Real-time SMS/push demand-response alerts

## Author
Solo project — 1M1B Applied AI for Climate Action internship.
