# Airline Flight Delay Prediction and Analysis Using Machine Learning

## Project Overview

An end-to-end machine learning project that predicts whether a scheduled
U.S. domestic flight will be significantly delayed (15+ minutes), using
only information available **before** the flight operates. The project
covers data exploration, cleaning, feature engineering, model comparison,
hyperparameter tuning, final evaluation, and a Streamlit app for interactive
predictions.

> ⚠️ **Honesty note (read before using):** This is a portfolio/educational
> project, not a production-ready delay predictor. As shown below, the final
> model's Recall is low (4.76%) — it misses most flights that are actually
> delayed. The dataset is also from 2015 and includes no real-time weather
> or air-traffic data. **Do not use this app to make real travel decisions.**

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
`DEPARTURE_DELAY`, `AIRLINE_DELAY`, `WEATHER_DELAY`, `AIR_SYSTEM_DELAY`,
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

## Model Comparison (Actual Results)

Trained on a stratified sample of 300,000 rows (see notebook Section 15),
target distribution: 81.39% Not Delayed / 18.61% Delayed.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| HistGradientBoosting | 0.8157 | 0.6519 | 0.0211 | 0.0409 | 0.6924 |
| Logistic Regression | 0.8139 | 0.5625 | 0.0008 | 0.0016 | 0.6347 |
| Random Forest | 0.8139 | 1.0000 | 0.0001 | 0.0002 | 0.6620 |

## Final Results

**Selected model:** HistGradientBoostingClassifier, tuned via `RandomizedSearchCV`
(best params: `max_iter=150, max_depth=6, learning_rate=0.2, l2_regularization=0.0`).

**Held-out test set performance (final, after tuning):**

| Metric | Value |
|---|---|
| Accuracy | 0.8172 |
| Precision | 0.6146 |
| Recall | 0.0476 |
| F1-score | 0.0883 |
| ROC-AUC | 0.7012 |

## Key Findings

HistGradientBoosting outperformed Logistic Regression and Random Forest on
F1-score and was selected as the final model. After tuning, it reached 81.7%
accuracy and a ROC-AUC of 0.70, showing the model ranks flights by delay
risk better than random chance. However, **Recall remained low at 4.76%**
even after tuning — the model correctly identifies most on-time flights but
catches fewer than 1 in 20 flights that are genuinely delayed. This is
largely driven by class imbalance (81.4% of flights are not delayed), which
means a naive "always predict not delayed" baseline would already score
close to 81% accuracy while catching zero actual delays. In practical terms,
pre-flight scheduling features (route, airline, time of day, distance)
provide a real but weak predictive signal — useful for ranking relative risk,
but not sufficient on their own to reliably flag individual at-risk flights.
Feature importance for the final model could not be extracted directly, since
`HistGradientBoostingClassifier` does not expose `feature_importances_`
(unlike Random Forest); permutation importance would be needed for that
analysis.

## Streamlit Demo

A local Streamlit app (`app/app.py`) lets you enter flight details and get
a delay prediction with probability, styled with a modern gradient/glass UI.
See [How to Run Streamlit](#how-to-run-streamlit) below.

*(If deployed to Streamlit Community Cloud or similar, add the live link here
— do not publish a fake link.)*

## Project Structure

```
airline-flight-delay-analysis/
├── data/
│   ├── raw/                  # airlines.csv, airports.csv, flights.csv 
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


## How to Run Notebook

```bash
jupyter notebook notebooks/airline_flight_delay_analysis.ipynb
```

or open it in Google Colab / VS Code's Jupyter extension. Run cells top to
bottom; each major section has Markdown explaining what it does and why.

Alternatively, run the headless training script:

```bash
python src/train_model.py --data data/raw/flights.csv --sample-size 300000
python src/evaluate_model.py --model models/best_model.pkl --data data/raw/flights.csv
```

## How to Run Streamlit

```bash
streamlit run app/app.py
```

Requires `models/best_model.pkl` to already exist (produced by training).

## Limitations

- **Low Recall (4.76%):** the model misses the large majority of genuinely
  delayed flights. It is not suitable for real-world delay-risk decisions
  in its current form.
- Dataset reflects 2015 flight operations; patterns may differ from current
  airline scheduling and operations.
- No real-time weather data is incorporated.
- No live air-traffic-control or airport-congestion data is incorporated.
- Predictions are probabilistic estimates, not guarantees.
- Categorical encoding caps airports to their 30 most frequent values;
  rarer airports are grouped into a single "infrequent" bucket, which may
  reduce accuracy for low-traffic routes specifically.
- Modeling was performed on a 300,000-row stratified sample of the full
  5.8M-row dataset for computational practicality (see notebook Section 15).

## Future Improvements

- Address class imbalance directly (e.g. class weighting, SMOTE, threshold
  tuning) to improve Recall
- Incorporate real-time weather data
- Incorporate airport congestion / air-traffic data
- Incorporate aircraft turnaround and live flight-status data
- Use more recent flight data (2015 → present)
- Explore additional boosting models and probability calibration
- Add permutation-based feature importance (since HistGradientBoosting has
  no native `feature_importances_`)
- Add explainable-AI (e.g. SHAP) analysis
- Deploy to a persistent cloud host