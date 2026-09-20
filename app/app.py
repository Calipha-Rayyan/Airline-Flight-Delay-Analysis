"""
Airline Flight Delay Predictor — Streamlit App

Run locally with:
    streamlit run app/app.py

This app loads the saved pipeline (preprocessing + model) produced by the
training notebook/script and uses the SAME feature-engineering functions
from src/feature_engineering.py, so predictions are consistent with training.
"""

import sys
import os
import joblib
import pandas as pd
import streamlit as st

# Allow importing from src/ regardless of where streamlit is launched from
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features, ALL_FEATURES  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
AIRLINES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "airlines.csv")
AIRPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "airports.csv")

st.set_page_config(page_title="Airline Flight Delay Predictor", page_icon="✈️", layout="centered")


@st.cache_resource
def load_pipeline(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_reference_data():
    airlines = pd.read_csv(AIRLINES_PATH)
    airports = pd.read_csv(AIRPORTS_PATH)
    return airlines, airports


pipeline = load_pipeline(MODEL_PATH)
airlines_df, airports_df = load_reference_data()

st.title("✈️ Airline Flight Delay Predictor")
st.write(
    "This application uses a machine-learning model trained on historical "
    "airline flight data to estimate whether a scheduled flight is likely "
    "to be delayed by 15 minutes or more."
)

if pipeline is None:
    st.warning(
        "No trained model found at `models/best_model.pkl`. "
        "Train the model first by running the notebook or "
        "`python src/train_model.py`, then reload this app."
    )
    st.stop()

st.subheader("Flight Details")

col1, col2 = st.columns(2)

with col1:
    airline_code = st.selectbox(
        "Airline",
        options=airlines_df["IATA_CODE"].tolist(),
        format_func=lambda code: f"{code} — {airlines_df.set_index('IATA_CODE').loc[code, 'AIRLINE']}",
    )
    origin_code = st.selectbox(
        "Origin Airport",
        options=airports_df["IATA_CODE"].tolist(),
        format_func=lambda code: f"{code} — {airports_df.set_index('IATA_CODE').loc[code, 'AIRPORT']}",
    )
    month = st.selectbox("Month", options=list(range(1, 13)), format_func=lambda m: pd.Timestamp(2015, m, 1).strftime("%B"))
    day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)

with col2:
    destination_code = st.selectbox(
        "Destination Airport",
        options=airports_df["IATA_CODE"].tolist(),
        index=1,
        format_func=lambda code: f"{code} — {airports_df.set_index('IATA_CODE').loc[code, 'AIRPORT']}",
    )
    day_of_week = st.selectbox(
        "Day of Week",
        options=[1, 2, 3, 4, 5, 6, 7],
        format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d - 1],
    )
    departure_time = st.time_input("Scheduled Departure Time")
    distance = st.number_input("Distance (miles)", min_value=1, max_value=6000, value=800)

scheduled_time = st.number_input(
    "Scheduled Flight Duration (minutes)", min_value=10, max_value=800, value=120,
    help="Total scheduled time in the air plus taxi, in minutes."
)

if origin_code == destination_code:
    st.error("Origin and destination airports must be different.")
    st.stop()

if st.button("Predict Flight Delay", type="primary"):
    scheduled_departure_int = departure_time.hour * 100 + departure_time.minute

    input_df = pd.DataFrame([{
        "AIRLINE": airline_code,
        "ORIGIN_AIRPORT": origin_code,
        "DESTINATION_AIRPORT": destination_code,
        "MONTH": month,
        "DAY": day,
        "DAY_OF_WEEK": day_of_week,
        "SCHEDULED_DEPARTURE": scheduled_departure_int,
        "SCHEDULED_TIME": scheduled_time,
        "DISTANCE": distance,
    }])

    try:
        input_engineered = engineer_features(input_df)
        X_input = input_engineered[ALL_FEATURES]

        prediction = pipeline.predict(X_input)[0]
        probability = pipeline.predict_proba(X_input)[0][1]

        st.subheader("Prediction")
        if prediction == 1:
            st.error(f"⚠️ Flight is likely to be **DELAYED**")
        else:
            st.success(f"✅ Flight is likely **NOT delayed**")

        st.metric("Probability of Delay", f"{probability * 100:.1f}%")

        with st.expander("Flight details used for this prediction"):
            st.write(f"**Airline:** {airline_code}")
            st.write(f"**Origin:** {origin_code}")
            st.write(f"**Destination:** {destination_code}")
            st.write(f"**Departure Hour:** {input_engineered['Departure_Hour'].iloc[0]}")
            st.write(f"**Time of Day:** {input_engineered['Time_of_Day'].iloc[0]}")
            st.write(f"**Distance:** {distance} miles ({input_engineered['Distance_Category'].iloc[0]})")
            st.write(f"**Weekend flight:** {'Yes' if input_engineered['Is_Weekend'].iloc[0] == 1 else 'No'}")

    except Exception as e:
        st.error(
            "Could not generate a prediction for these inputs. This can happen "
            "if the airline or airport code was not present in the training data."
        )
        st.exception(e)

st.divider()
st.caption(
    "Model trained on 2015 U.S. domestic flight data (DOT/BTS via Kaggle). "
    "Predictions are probabilistic estimates based on historical patterns, "
    "not guarantees, and do not account for real-time weather or air-traffic conditions."
)
