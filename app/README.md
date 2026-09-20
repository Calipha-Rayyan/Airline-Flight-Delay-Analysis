# ✈️ Airline Flight Delay Predictor

A complete machine-learning project for **airline flight-delay analysis and prediction**.

The project uses historical U.S. domestic flight data to estimate whether a scheduled flight is likely to experience a significant arrival delay. It covers the full workflow from data exploration and feature engineering to model training, evaluation, model persistence, and an interactive Streamlit application.

> **Portfolio / Educational Project:** The predictions are based on historical 2015 flight data. The application does not use live weather, air-traffic, airport-congestion, or real-time flight-status information and should not be used for real travel decisions.

---

## 📌 Project Overview

### Problem

Flight delays create operational and passenger-planning challenges. Historical flight records contain patterns related to airlines, airports, scheduled departure times, distance, dates, and other flight characteristics.

This project asks:

> **Can we predict whether a scheduled flight will be delayed by 15 minutes or more using information available before the flight operates?**

### Prediction Task

This is a **binary classification** problem:

| Target | Meaning |
|---|---|
| `0` | Not significantly delayed |
| `1` | Arrival delay of 15+ minutes |

The target is derived from the historical `ARRIVAL_DELAY` field during model development. The actual arrival-delay value is then excluded from model inputs to avoid target leakage.

---

## 🗂️ Dataset

**Dataset:** 2015 Flight Delays and Cancellations

**Source:** U.S. Department of Transportation / Bureau of Transportation Statistics, distributed through Kaggle.

**Kaggle:**  
https://www.kaggle.com/datasets/usdot/flight-delays

The dataset contains approximately **5.8 million flight records** and includes:

- `flights.csv`
- `airlines.csv`
- `airports.csv`

The main modeling data comes from `flights.csv`.

### Data Leakage Prevention

Because the application is intended to estimate delay risk before a flight operates, post-flight variables are not used as prediction inputs.

Excluded leakage / post-flight fields include:

```text
ARRIVAL_DELAY
DEPARTURE_DELAY
CARRIER_DELAY
WEATHER_DELAY
NAS_DELAY
SECURITY_DELAY
LATE_AIRCRAFT_DELAY
```

The model instead relies on information that can reasonably be available before departure.

---

## 🧠 Machine-Learning Pipeline

```text
Historical Flight Data
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
Target Creation
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Train / Test Split
        ↓
Preprocessing Pipeline
        ↓
Candidate Models
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Final Evaluation
        ↓
Save Preprocessing + Model Pipeline
        ↓
Streamlit Prediction App
```

---

## 🛠️ Technology Stack

### Data & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Visualization

- Matplotlib
- Seaborn

### Application

- Streamlit
- HTML/CSS styling through Streamlit

### Development

- Jupyter Notebook / Google Colab
- VS Code
- Git / GitHub

---

## ⚙️ Feature Engineering

The project creates additional features designed to capture useful flight-time and scheduling patterns.

### `Departure_Hour`

Extracts the scheduled departure hour from `SCHEDULED_DEPARTURE`.

Example:

```text
1744 → 17
```

### `Departure_Minute`

Extracts the scheduled departure minute.

Example:

```text
1744 → 44
```

### `Is_Weekend`

Identifies whether the scheduled flight falls on Saturday or Sunday.

```text
0 → Weekday
1 → Weekend
```

### `Time_of_Day`

Converts scheduled departure time into a broader time category such as:

```text
Morning
Afternoon
Evening
Night
```

### `Distance_Category`

Groups flight distance into meaningful categories used by the feature-engineering pipeline.

The exact feature transformations are maintained centrally in:

```text
src/feature_engineering.py
```

The same feature-engineering logic is used by the training pipeline and the Streamlit application.

---

## 🤖 Machine-Learning Models

The project compares multiple classification approaches:

### 1. Logistic Regression

Used as a simple baseline model.

### 2. Random Forest Classifier

Used to model nonlinear relationships and provide tree-based feature-importance analysis.

### 3. HistGradientBoostingClassifier

Used as a stronger tabular-data classifier suitable for large datasets.

The final model is selected using actual validation/test results rather than assuming a model will perform best in advance.

---

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The project pays particular attention to **Recall**, because a model can have reasonable overall accuracy while still missing many genuinely delayed flights.

The notebook contains the actual metric values and model comparison.

> **Do not treat accuracy alone as sufficient evidence of model quality.** The class distribution and the precision/recall trade-off must also be considered.

---

## 📈 Visualizations

The project includes visual analysis such as:

- Target/class distribution
- Delay rate by departure hour
- Model comparison
- Confusion matrix
- ROC curve
- Feature-importance chart

These visualizations are generated from the actual model/data outputs and are stored in the `results/` directory where applicable.

---

## 💾 Saved Model Pipeline

The application loads:

```text
models/best_model.pkl
```

This saved artifact contains the trained prediction pipeline produced by the training workflow.

The Streamlit application does **not** create a separate preprocessing implementation. It uses the same feature-engineering functions from:

```text
src/feature_engineering.py
```

and the same saved preprocessing/model pipeline used by training.

This helps keep training-time and application-time transformations consistent.

---

# 🖥️ Streamlit Application

The project includes an interactive web application:

> **Airline Flight Delay Predictor**

The app allows a user to enter scheduled flight information and receive a delay-risk estimate.

### User Inputs

The application currently provides inputs for:

- Airline
- Origin Airport
- Destination Airport
- Month
- Day of Month
- Day of Week
- Scheduled Departure Time
- Distance
- Scheduled Flight Duration

The application automatically recomputes the engineered features required by the model.

### Prediction Output

The application displays:

- Predicted class:
  - `Delayed`
  - `Not Delayed`
- Probability of delay
- Flight summary
- Model-information note

---

## 🎨 Streamlit UI/UX

The application has been designed as a modern dark AI/aviation dashboard rather than a default Streamlit form.

Current presentation features include:

- Dark navy/charcoal visual theme
- Indigo/violet primary accents
- Cyan accent color
- High-contrast text
- Responsive two-column flight form
- Rounded cards
- Subtle shadows
- Animated hero entrance
- Button hover/active effects
- Result-card animations
- Animated delay-probability bar
- Accessible reduced-motion fallback
- Improved input focus states
- Styled dropdown controls
- Clear semantic styling for delayed vs. on-time predictions

### Dropdown Visibility

Streamlit's `selectbox` controls use BaseWeb components, so the closed input and the opened dropdown menu require separate styling.

The application explicitly styles:

```text
[data-baseweb="select"]
[data-baseweb="popover"]
[data-baseweb="menu"]
```

to improve readability of:

- Selected values
- Dropdown options
- Hover states
- Selected states
- Focus states

This avoids relying on generic CSS selectors that do not match Streamlit's actual dropdown structure.

---

## 📁 Project Structure

```text
airline-flight-delay-analysis/
│
├── data/
│   ├── raw/
│   │   ├── flights.csv
│   │   ├── airlines.csv
│   │   └── airports.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── airline_flight_delay_analysis.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── target_distribution.png
│
├── app/
│   ├── app.py
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

> The exact files may vary depending on which optional components have been generated. The raw 5.8M-row dataset should not be committed directly to GitHub.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd airline-flight-delay-analysis
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Add the Dataset

Download the dataset from:

https://www.kaggle.com/datasets/usdot/flight-delays

Place the files in:

```text
data/raw/
```

Expected:

```text
data/raw/flights.csv
data/raw/airlines.csv
data/raw/airports.csv
```

Because `flights.csv` is very large, keep the raw dataset outside GitHub.

---

# 📓 Running the Notebook

Open:

```text
notebooks/airline_flight_delay_analysis.ipynb
```

The notebook covers the complete machine-learning workflow:

```text
EDA
→ Cleaning
→ Target Creation
→ Feature Engineering
→ Train/Test Split
→ Preprocessing
→ Model Training
→ Model Comparison
→ Hyperparameter Tuning
→ Final Evaluation
→ Feature Importance
→ Model Saving
```

### Google Colab

The project can also be developed in Google Colab.

A recommended setup is:

```text
Google Drive
└── airline-flight-delay-analysis/
    ├── data/
    ├── notebooks/
    ├── models/
    └── results/
```

For a very large dataset, avoid repeatedly uploading `flights.csv` through the browser. Store the dataset in Google Drive or use another suitable persistent storage method and connect it to the Colab runtime.

---

# 🌐 Running the Streamlit App

The application requires the trained model:

```text
models/best_model.pkl
```

This model is created by the training notebook or training script.

From the project root:

```bash
streamlit run app/app.py
```

Streamlit will print a local address, typically:

```text
http://localhost:8501
```

Open that address in your browser.

---

## Alternative: Train the Model from the Command Line

If the training script is available in your checkout, it can be run with:

```bash
python src/train_model.py --data data/raw/flights.csv --sample-size 1000000
```

This uses a practical sample size for large-data experimentation.

Adjust the sample size according to available RAM and compute resources.

---

# 🔬 Why Sampling May Be Used

The source dataset contains roughly 5.8 million records.

The full dataset is valuable for exploration, but training every model and running large hyperparameter searches on all records can consume substantial RAM and time.

A reproducible training sample may therefore be used when appropriate.

The project should document:

- original dataset size
- modeling sample size
- reason for sampling
- random seed
- class distribution

Example random state:

```python
RANDOM_STATE = 42
```

Sampling is a computational decision, not a replacement for proper test-set evaluation.

---

# 🔐 Data Leakage Prevention

A major design goal is to make the prediction realistic.

The application only accepts information that can reasonably be available before the flight operates.

For example:

```text
Airline
Origin
Destination
Month
Day
Day of Week
Scheduled Departure
Scheduled Duration
Distance
```

It does not ask the user for:

```text
Actual Arrival Delay
Actual Departure Delay
Weather Delay
Carrier Delay
NAS Delay
Security Delay
Late Aircraft Delay
```

This is essential because using post-flight outcomes as input would leak the answer into the model.

---

# ⚠️ Model Limitations

The application is intentionally presented as a portfolio/educational system.

Important limitations include:

1. The model was trained using **2015 U.S. domestic flight data**.
2. Airline operations and delay patterns can change over time.
3. The model does not use real-time weather data.
4. The model does not use live air-traffic conditions.
5. The model does not use airport-congestion information.
6. Predictions are probabilistic estimates rather than guarantees.
7. The underlying model has limited recall, so it can miss genuinely delayed flights.

The app should therefore **not be used for real travel decisions**.

---

# 🔮 Future Improvements

Possible future extensions include:

- More recent flight datasets
- Real-time weather integration
- Airport-congestion information
- Live air-traffic data
- Aircraft turnaround information
- Flight-status data
- Model calibration
- More advanced gradient-boosting models
- Explainable AI techniques
- Model monitoring
- Cloud deployment
- Automated retraining

---

# 👨‍💻 Project Purpose

This project demonstrates practical skills in:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Classification
- Model comparison
- Hyperparameter tuning
- Model evaluation
- Feature-importance analysis
- Model persistence
- Streamlit application development
- Git/GitHub project organization

It is intended to demonstrate an **end-to-end machine-learning workflow**, not simply a collection of charts or a single trained model.

---

# 📜 License / Dataset Notice

The dataset is obtained from the public Kaggle distribution of U.S. Department of Transportation / Bureau of Transportation Statistics flight data.

Review the original dataset page for the applicable data/source terms:

https://www.kaggle.com/datasets/usdot/flight-delays

---

## ⭐ Project Status

**Core ML pipeline:** Implemented  
**Data analysis:** Implemented through the project notebook  
**Feature engineering:** Implemented  
**Model comparison:** Implemented through the training workflow  
**Hyperparameter tuning:** Implemented through the training workflow  
**Saved model pipeline:** Implemented  
**Streamlit application:** Implemented  
**UI/UX refinement:** Implemented  
**Public deployment:** Add the live URL here once deployed

---

## 👤 Author

**Muhammad Rayyan Bhatti**

Machine Learning / Data Science Portfolio Project
