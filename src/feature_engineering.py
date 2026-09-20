"""
feature_engineering.py

Shared feature-engineering logic used by BOTH the training notebook and the
Streamlit app. Keeping this in one place guarantees the app applies the exact
same transformations the model was trained on (no train/serve skew).

All functions operate on pre-flight information only. Nothing here touches
ARRIVAL_DELAY, DEPARTURE_DELAY, or any *_DELAY cause column.
"""

import pandas as pd
import numpy as np


def extract_departure_hour(scheduled_departure: pd.Series) -> pd.Series:
    """
    SCHEDULED_DEPARTURE is stored as an integer/string like 1830 meaning 18:30.
    Extract the hour component (0-23).
    """
    sd = scheduled_departure.astype(int).astype(str).str.zfill(4)
    return sd.str[:2].astype(int)


def extract_departure_minute(scheduled_departure: pd.Series) -> pd.Series:
    """
    Extract the minute component (0-59) from the same HHMM integer format.
    """
    sd = scheduled_departure.astype(int).astype(str).str.zfill(4)
    return sd.str[2:4].astype(int)


def compute_is_weekend(day_of_week: pd.Series) -> pd.Series:
    """
    DAY_OF_WEEK in this dataset is 1=Monday ... 7=Sunday.
    Weekend = Saturday(6) or Sunday(7).
    """
    return day_of_week.isin([6, 7]).astype(int)


def compute_time_of_day(departure_hour: pd.Series) -> pd.Series:
    """
    Bucket departure hour into a human-readable time-of-day category.

    Thresholds (24h clock):
        Night:      00:00 - 04:59
        Morning:    05:00 - 11:59
        Afternoon:  12:00 - 16:59
        Evening:    17:00 - 23:59
    """
    bins = [-1, 4, 11, 16, 23]
    labels = ["Night", "Morning", "Afternoon", "Evening"]
    return pd.cut(departure_hour, bins=bins, labels=labels).astype(str)


def compute_distance_category(distance: pd.Series) -> pd.Series:
    """
    Bucket flight distance (miles) into Short / Medium / Long.

    Thresholds chosen from common aviation conventions:
        Short:  < 500 miles   (regional hops)
        Medium: 500-1500 miles (typical domestic routes)
        Long:   > 1500 miles  (transcontinental / long-haul domestic)
    """
    bins = [-1, 500, 1500, np.inf]
    labels = ["Short", "Medium", "Long"]
    return pd.cut(distance, bins=bins, labels=labels).astype(str)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all engineered features to a dataframe containing at minimum:
        SCHEDULED_DEPARTURE, DAY_OF_WEEK, DISTANCE

    Returns a copy with new columns added:
        Departure_Hour, Departure_Minute, Is_Weekend, Time_of_Day, Distance_Category
    """
    out = df.copy()

    out["Departure_Hour"] = extract_departure_hour(out["SCHEDULED_DEPARTURE"])
    out["Departure_Minute"] = extract_departure_minute(out["SCHEDULED_DEPARTURE"])
    out["Is_Weekend"] = compute_is_weekend(out["DAY_OF_WEEK"])
    out["Time_of_Day"] = compute_time_of_day(out["Departure_Hour"])
    out["Distance_Category"] = compute_distance_category(out["DISTANCE"])

    return out


# Feature lists used consistently across notebook, training script, and app.

NUMERIC_FEATURES = [
    "DISTANCE",
    "SCHEDULED_TIME",
    "Departure_Hour",
    "Departure_Minute",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
]

CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "Is_Weekend",
    "Time_of_Day",
    "Distance_Category",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must NEVER be used as model inputs (post-flight / leakage).
# Verified against the actual dataset schema (this Kaggle release uses
# AIR_SYSTEM_DELAY and AIRLINE_DELAY rather than NAS_DELAY/CARRIER_DELAY).
LEAKAGE_COLUMNS = [
    "ARRIVAL_DELAY",
    "DEPARTURE_DELAY",
    "AIRLINE_DELAY",
    "WEATHER_DELAY",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "ARRIVAL_TIME",
    "DEPARTURE_TIME",
    "TAXI_IN",
    "TAXI_OUT",
    "WHEELS_ON",
    "WHEELS_OFF",
    "AIR_TIME",
    "ELAPSED_TIME",
    "CANCELLED",
    "CANCELLATION_REASON",
    "DIVERTED",
]