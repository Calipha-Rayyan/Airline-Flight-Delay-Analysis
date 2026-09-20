# Airline Flight Delay Prediction and Analysis Using Machine Learning

## Project Overview

An end-to-end machine learning project that predicts whether a scheduled
U.S. domestic flight will be significantly delayed (15+ minutes), using
only information available **before** the flight operates. The project
covers data exploration, cleaning, feature engineering, model comparison,
hyperparameter tuning, final evaluation, and a Streamlit app for interactive
predictions.

## Problem Statement

Flight delays cost airlines, airports, and passengers time and money.
Being able to flag high-risk flights *before departure* — using only
schedule-level information — could help with staffing, rebooking policy,
and passenger communication.

## Objective

> Can we predict whether a scheduled flight will be delayed using
> information available before the flight operates?

This is framed as a binary classification problem:

```
Delayed = 1 if ARRIVAL_DELAY >= 15 minutes, else 0
```

## Dataset

**2015 Flight Delays and Cancellations** — U.S. DOT / Bureau of Transportation
Statistics, via Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays

- `flights.csv` — ~5.8 million flight records (included)
- `airlines.csv` — airline code lookup (included)
- `airports.csv` — airport code lookup (included)

## Data Cleaning

- Removed exact duplicate rows
- Removed cancelled and diverted flights (no valid arrival-delay target)
- Removed rows with missing `ARRIVAL_DELAY` (target cannot be computed)
- Validated `SCHEDULED_DEPARTURE` timestamps (must be valid HHMM values)
- No outlier removal on delay magnitude — long legitimate delays are kept

All cleaning steps are logged with row counts and reasons; see
`src/data_cleaning.py` and the notebook's "Data Cleaning" section.

## Data Leakage Prevention

The target is derived from `ARRIVAL_DELAY`, but the following post-flight
columns are **excluded from model inputs**: `ARRIVAL_DELAY`,
`DEPARTURE_DELAY`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`,
`SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`, and other post-flight timing fields
(see `LEAKAGE_COLUMNS` in `src/feature_engineering.py`). Only information
known at scheduling time is used as input.

## Feature Engineering

| Feature | Description |
|---|---|
| `Departure_Hour` | Hour extracted from `SCHEDULED_DEPARTURE` |
| `Departure_Minute` | Minute extracted from `SCHEDULED_DEPARTURE` |
| `Is_Weekend` | 1 if Saturday/Sunday, else 0 |
| `Time_of_Day` | Night / Morning / Afternoon / Evening bucket |
| `Distance_Category` | Short (<500mi) / Medium (500-1500mi) / Long (>1500mi) |

## Models Compared

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest Classifier | Nonlinear model, feature importance |
| HistGradientBoostingClassifier | Strong tabular model for large data |

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, and ROC-AUC are reported for every
model. **Recall** is weighted heavily in model selection, since missing a
genuinely delayed flight has real operational cost.

## Model Comparison

*(Populated after running the notebook — see `results/model_comparison.csv`.
Values below are placeholders and must be replaced with actual output.)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| HistGradientBoosting | — | — | — | — | — |

## Final Results

*(To be filled in after final held-out test evaluation — do not report
until the notebook has actually been executed.)*

## Key Findings

*(Write 3–5 sentences here after running the full pipeline, based only on
actual results: best-performing model, important predictive features,
effect of class imbalance, and what the confusion matrix / F1 / ROC-AUC
mean in this context.)*

## Streamlit Demo

A local Streamlit app (`app/app.py`) lets you enter flight details and get
a delay prediction with probability. See [How to Run Streamlit](#how-to-run-streamlit)
below.

*(If deployed to Streamlit Community Cloud or similar, add the live link here
— do not publish a fake link.)*

## Project Structure

```
airline-flight-delay-analysis/
├── data/
│   ├── raw/                  # airlines.csv, airports.csv, flights.csv — see data/README.md
│   └── processed/            # cleaned/engineered data (generated)
├── notebooks/
│   └── airline_flight_delay_analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/
│   └── best_model.pkl        # generated after training
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── target_distribution.png
├── app/
│   ├── app.py
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone <your-repo-url>
cd airline-flight-delay-analysis
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then download `flights.csv` per `data/README.md` before running the notebook.

## How to Run Notebook

```bash
jupyter notebook notebooks/airline_flight_delay_analysis.ipynb
```

or open it in Google Colab / VS Code's Jupyter extension. Run cells top to
bottom; each major section has Markdown explaining what it does and why.

Alternatively, run the headless training script:

```bash
python src/train_model.py --data data/raw/flights.csv --sample-size 1000000
python src/evaluate_model.py --model models/best_model.pkl --data data/raw/flights.csv
```

## How to Run Streamlit

```bash
streamlit run app/app.py
```

Requires `models/best_model.pkl` to already exist (produced by training).

## Limitations

- Dataset reflects 2015 flight operations; patterns may differ from current
  airline scheduling and operations.
- No real-time weather data is incorporated.
- No live air-traffic-control or airport-congestion data is incorporated.
- Predictions are probabilistic estimates, not guarantees.
- Some features used here (e.g. historical route-level delay patterns) may
  be harder to obtain in a live production system.
- Modeling was performed on a reproducible sample of the full 5.8M-row
  dataset for computational practicality (see notebook for exact sample size
  and method).

## Future Improvements

- Incorporate real-time weather data
- Incorporate airport congestion / air-traffic data
- Incorporate aircraft turnaround and live flight-status data
- Use more recent flight data (2015 → present)
- Explore additional boosting models and calibration
- Add explainable-AI (e.g. SHAP) analysis
- Deploy to a persistent cloud host
