# Streamlit App — Airline Flight Delay Predictor

A simple web interface for the trained flight-delay model.

## Prerequisites

The trained pipeline must exist at `../models/best_model.pkl`. This is
produced by running the notebook (`notebooks/airline_flight_delay_analysis.ipynb`)
through the "Save Model" section, or by running:

```bash
python src/train_model.py --data data/raw/flights.csv --sample-size 1000000
```

from the project root.

## Running locally

From the project root:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## What it does

1. Loads the saved preprocessing + model pipeline.
2. Presents a form for airline, origin/destination airport, date, scheduled
   departure time, and distance.
3. Internally recomputes the same engineered features used at training time
   (`Departure_Hour`, `Departure_Minute`, `Is_Weekend`, `Time_of_Day`,
   `Distance_Category`) via `src/feature_engineering.py` — no separate/divergent
   preprocessing logic.
4. Displays the predicted class (Delayed / Not Delayed) and the predicted
   probability of delay.

## Notes

- If an airline or airport code wasn't seen during training, the app relies
  on `OneHotEncoder(handle_unknown="ignore")` in the saved pipeline to handle
  it gracefully rather than crashing.
- This app is for demonstration purposes. It does not incorporate real-time
  weather, air-traffic, or airport-congestion data.
