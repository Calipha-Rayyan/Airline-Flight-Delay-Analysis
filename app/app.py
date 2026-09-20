"""
Airline Flight Delay Predictor — Streamlit App

Run locally with:
    streamlit run app/app.py

This app loads the saved pipeline (preprocessing + model) produced by the
training notebook/script and uses the SAME feature-engineering functions
from src/feature_engineering.py, so predictions are consistent with training.
No model, preprocessing, or feature-engineering logic is changed here —
this file only affects presentation (layout, CSS, copy).

IMPORTANT — Honesty note:
This is a portfolio/educational demonstration. The underlying model has low
Recall (see README), was trained on 2015 data, and has no access to
real-time weather or air-traffic conditions. It should NOT be used to make
real travel decisions.
"""

import sys
import os
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import engineer_features, ALL_FEATURES  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
AIRLINES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "airlines.csv")
AIRPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "airports.csv")

st.set_page_config(
    page_title="Airline Flight Delay Predictor",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==========================================================================
# STYLING
#
# Design notes for future maintainers:
# - Palette: dark navy/charcoal base, indigo/violet primary, cyan accent,
#   semantic green/amber/red for results. No teal, no low-contrast gray text.
# - Cards use Streamlit's native st.container(border=True) rather than
#   hand-rolled <div>...</div> pairs split across separate st.markdown()
#   calls. The old version opened a styled <div class="glass-card"> in one
#   st.markdown() call and closed it in a later, separate call. Streamlit
#   wraps every st.markdown() output in its own isolated block, so the
#   "opening" div rendered as a standalone, empty, padded box with nothing
#   inside it — this was the unexplained empty rectangle under the hero.
#   Using st.container(border=True) avoids this failure mode entirely: it's
#   a single real Streamlit container, styled via the stable
#   [data-testid="stVerticalBlockBorderWrapper"] selector.
# - Streamlit's selectbox is a BaseWeb component. The closed control and the
#   open dropdown menu are separate DOM subtrees (the menu is teleported to
#   a portal). Both are targeted explicitly below via [data-baseweb="select"]
#   and [data-baseweb="popover"]/[data-baseweb="menu"], rather than generic
#   `select { color: white }` rules that do not match Streamlit's actual
#   markup and would silently do nothing.
# ==========================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

:root {
    --bg-base: #0a0e1a;
    --bg-surface: #121729;
    --bg-surface-hover: #171d33;
    --border-subtle: rgba(148, 163, 184, 0.16);
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --primary: #6366f1;
    --primary-light: #818cf8;
    --secondary: #8b5cf6;
    --accent: #22d3ee;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
}

/* ---------- Page background: dark navy with subtle glow, no flat teal ---------- */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.16) 0%, transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(139, 92, 246, 0.14) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(34, 211, 238, 0.06) 0%, transparent 50%),
        var(--bg-base);
}

.block-container {
    padding-top: 2rem;
    max-width: 780px;
}

/* ---------- Respect reduced-motion preference ---------- */
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}

/* ---------- Hero ---------- */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.4);
    color: var(--primary-light);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}

.hero-title {
    color: var(--text-primary);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.6rem 0;
    line-height: 1.15;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.02rem;
    line-height: 1.55;
    max-width: 560px;
    margin: 0;
}

.hero-wrap {
    padding: 2.4rem 2rem 2.2rem 2rem;
    border-radius: 20px;
    background: linear-gradient(160deg, rgba(99,102,241,0.14) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(129, 140, 248, 0.22);
    margin-bottom: 1.6rem;
    animation: fadeInDown 0.6s ease-out;
}

/* ---------- Card containers (native Streamlit bordered containers) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 18px !important;
    padding: 0.4rem 0.4rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    animation: fadeIn 0.5s ease-out;
}

.card-title {
    color: var(--text-primary);
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-subtitle {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-bottom: 1.1rem;
}

/* ---------- Widget labels ---------- */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}

/* ---------- Selectbox: closed control ---------- */
[data-baseweb="select"] > div {
    background-color: #0e1324 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    transition: border-color 200ms ease, box-shadow 200ms ease;
}

[data-baseweb="select"] > div:hover {
    border-color: rgba(129, 140, 248, 0.5) !important;
}

[data-baseweb="select"]:focus-within > div {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18) !important;
}

/* Selected value text + placeholder inside the closed control */
[data-baseweb="select"] div,
[data-baseweb="select"] span {
    color: var(--text-primary) !important;
}

/* Dropdown arrow icon */
[data-baseweb="select"] svg {
    fill: var(--text-secondary) !important;
}

/* ---------- Selectbox: open menu (rendered in a portal) ---------- */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #141a30 !important;
    border: 1px solid rgba(129, 140, 248, 0.35) !important;
    border-radius: 12px !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.5) !important;
    overflow: hidden;
}

[data-baseweb="popover"] [data-baseweb="menu"] li,
[data-baseweb="popover"] ul[role="listbox"] li {
    color: var(--text-primary) !important;
    background-color: transparent !important;
    font-size: 0.92rem !important;
}

[data-baseweb="popover"] [data-baseweb="menu"] li:hover,
[data-baseweb="popover"] ul[role="listbox"] li:hover {
    background-color: rgba(99, 102, 241, 0.28) !important;
    color: #ffffff !important;
}

[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: rgba(99, 102, 241, 0.4) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ---------- Number & time inputs ---------- */
[data-testid="stNumberInput"] input,
[data-testid="stTimeInput"] input {
    background-color: #0e1324 !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    transition: border-color 200ms ease, box-shadow 200ms ease;
}

[data-testid="stNumberInput"] input:focus,
[data-testid="stTimeInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18) !important;
}

[data-testid="stNumberInput"] button {
    background-color: #171d33 !important;
    border-color: var(--border-subtle) !important;
    color: var(--text-secondary) !important;
}

/* ---------- Predict button ---------- */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #22d3ee 100%);
    background-size: 160% 160%;
    color: white;
    font-weight: 700;
    font-size: 1.02rem;
    padding: 0.8rem 1rem;
    border: none;
    border-radius: 12px;
    box-shadow: 0 10px 24px rgba(99, 102, 241, 0.35);
    transition: transform 180ms ease, box-shadow 180ms ease, background-position 400ms ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 32px rgba(99, 102, 241, 0.5);
    background-position: 100% 0%;
    color: white;
}

div.stButton > button:active {
    transform: translateY(0px) scale(0.99);
}

/* ---------- Result card semantic accents (targeted via marker class) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.marker-delayed) {
    border-color: rgba(239, 68, 68, 0.45) !important;
    background: linear-gradient(160deg, rgba(239,68,68,0.10) 0%, var(--bg-surface) 55%) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(.marker-ontime) {
    border-color: rgba(34, 197, 94, 0.45) !important;
    background: linear-gradient(160deg, rgba(34,197,94,0.10) 0%, var(--bg-surface) 55%) !important;
}

.result-title {
    font-size: 1.4rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    animation: popIn 0.4s ease-out;
}

.result-title-delayed { color: #fca5a5; }
.result-title-ontime { color: #86efac; }

.result-desc {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-bottom: 1.1rem;
}

.prob-number {
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}

.prob-caption {
    color: var(--text-muted);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}

.prob-bar-bg {
    background: rgba(148, 163, 184, 0.15);
    border-radius: 999px;
    height: 14px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 700ms cubic-bezier(0.16, 1, 0.3, 1);
}

/* ---------- Summary table rows ---------- */
.summary-row {
    display: flex;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.92rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-label { color: var(--text-muted); font-weight: 500; }
.summary-value { color: var(--text-primary); font-weight: 600; text-align: right; }

/* ---------- Model info card ---------- */
.info-icon-row {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
}
.info-text {
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    line-height: 1.6;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-subtle);
}
.app-footer strong { color: var(--text-secondary); }

/* ---------- Expander ---------- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background-color: var(--bg-surface) !important;
    border-radius: 10px !important;
    color: var(--text-secondary) !important;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(0.94); }
    100% { opacity: 1; transform: scale(1); }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# DATA / MODEL LOADING (unchanged logic — presentation-only refactor)
# ==========================================================================

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


# ==========================================================================
# RENDER HELPERS
# ==========================================================================

def render_header():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge">🛰️ AI-POWERED FLIGHT RISK ANALYSIS</div>
            <div class="hero-title">✈️ Airline Flight Delay Predictor</div>
            <p class="hero-subtitle">
                Estimate delay risk using historical flight patterns and machine learning.
                Enter a scheduled flight below to get an instant risk estimate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flight_form(airlines_df, airports_df):
    with st.container(border=True):
        st.markdown('<div class="card-title">🧭 Flight Details</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-subtitle">Enter the scheduled flight information to estimate delay risk.</div>',
            unsafe_allow_html=True,
        )

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
            month = st.selectbox(
                "Month", options=list(range(1, 13)),
                format_func=lambda m: pd.Timestamp(2015, m, 1).strftime("%B"),
            )
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
                format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                        "Friday", "Saturday", "Sunday"][d - 1],
            )
            departure_time = st.time_input("Scheduled Departure Time")
            distance = st.number_input("Distance (miles)", min_value=1, max_value=6000, value=800)

        scheduled_time = st.number_input(
            "Scheduled Flight Duration (minutes)", min_value=10, max_value=800, value=120,
            help="Total scheduled time in the air plus taxi, in minutes.",
        )

        if origin_code == destination_code:
            st.error("⚠️ Origin and destination airports must be different.")
            st.stop()

        predict_clicked = st.button("✈️  Predict Flight Delay")

    return {
        "airline_code": airline_code,
        "origin_code": origin_code,
        "destination_code": destination_code,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "departure_time": departure_time,
        "distance": distance,
        "scheduled_time": scheduled_time,
    }, predict_clicked


def render_result(pipeline, inputs, airlines_df, airports_df):
    departure_time = inputs["departure_time"]
    scheduled_departure_int = departure_time.hour * 100 + departure_time.minute

    input_df = pd.DataFrame([{
        "AIRLINE": inputs["airline_code"],
        "ORIGIN_AIRPORT": inputs["origin_code"],
        "DESTINATION_AIRPORT": inputs["destination_code"],
        "MONTH": inputs["month"],
        "DAY": inputs["day"],
        "DAY_OF_WEEK": inputs["day_of_week"],
        "SCHEDULED_DEPARTURE": scheduled_departure_int,
        "SCHEDULED_TIME": inputs["scheduled_time"],
        "DISTANCE": inputs["distance"],
    }])

    try:
        with st.spinner("Analyzing flight pattern..."):
            input_engineered = engineer_features(input_df)
            X_input = input_engineered[ALL_FEATURES]
            prediction = pipeline.predict(X_input)[0]
            probability = pipeline.predict_proba(X_input)[0][1]

        prob_pct = probability * 100

        with st.container(border=True):
            if prediction == 1:
                st.markdown('<div class="marker-delayed" style="display:none"></div>', unsafe_allow_html=True)
                st.markdown('<div class="result-title result-title-delayed">⚠️ DELAY RISK DETECTED</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="result-desc">This flight is predicted to be delayed by 15+ minutes.</div>',
                            unsafe_allow_html=True)
                bar_gradient = "linear-gradient(90deg, #f59e0b 0%, #ef4444 100%)"
            else:
                st.markdown('<div class="marker-ontime" style="display:none"></div>', unsafe_allow_html=True)
                st.markdown('<div class="result-title result-title-ontime">✓ LOW DELAY RISK</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="result-desc">This flight is predicted not to be delayed.</div>',
                            unsafe_allow_html=True)
                bar_gradient = "linear-gradient(90deg, #22c55e 0%, #38ef7d 100%)"

            st.markdown('<div class="prob-caption">Probability of Delay</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prob-number">{prob_pct:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob_pct:.1f}%; background:{bar_gradient};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---- Flight summary card (friendly names, only fields actually used) ----
        airline_name = airlines_df.set_index("IATA_CODE").loc[inputs["airline_code"], "AIRLINE"]
        origin_name = inputs["origin_code"]
        dest_name = inputs["destination_code"]

        with st.container(border=True):
            st.markdown('<div class="card-title">📋 Flight Summary</div>', unsafe_allow_html=True)
            rows = [
                ("Airline", airline_name),
                ("Route", f"{origin_name} → {dest_name}"),
                ("Departure", departure_time.strftime("%H:%M")),
                ("Time of Day", input_engineered["Time_of_Day"].iloc[0]),
                ("Distance", f"{inputs['distance']} miles ({input_engineered['Distance_Category'].iloc[0]})"),
                ("Weekend Flight", "Yes" if input_engineered["Is_Weekend"].iloc[0] == 1 else "No"),
            ]
            rows_html = "".join(
                f'<div class="summary-row"><span class="summary-label">{label}</span>'
                f'<span class="summary-value">{value}</span></div>'
                for label, value in rows
            )
            st.markdown(rows_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(
            "Could not generate a prediction for these inputs. This can happen "
            "if the airline or airport code was not present in the training data."
        )
        st.exception(e)


def render_model_info():
    with st.container(border=True):
        st.markdown(
            """
            <div class="info-icon-row">
                <div style="font-size:1.3rem;">ℹ️</div>
                <div>
                    <div class="card-title" style="margin-bottom:0.3rem;">About This Model</div>
                    <p class="info-text">
                        This application uses a machine-learning model trained on historical
                        U.S. domestic flight data (2015, DOT/BTS via Kaggle). Predictions are
                        estimates based on learned historical patterns and do not include
                        real-time weather or air-traffic conditions. This is an educational
                        portfolio project — not a real travel-planning tool.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <strong>Airline Flight Delay Predictor</strong><br>
            Built with Streamlit · Python · Scikit-learn<br>
            Machine Learning Portfolio Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    render_header()

    pipeline = load_pipeline(MODEL_PATH)
    airlines_df, airports_df = load_reference_data()

    if pipeline is None:
        st.warning(
            "No trained model found at `models/best_model.pkl`. "
            "Train the model first by running the notebook or "
            "`python src/train_model.py`, then reload this app."
        )
        st.stop()

    inputs, predict_clicked = render_flight_form(airlines_df, airports_df)

    if predict_clicked:
        render_result(pipeline, inputs, airlines_df, airports_df)

    render_model_info()
    render_footer()


if __name__ == "__main__":
    main()