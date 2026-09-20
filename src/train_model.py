"""
train_model.py

Standalone script version of the training pipeline. The notebook is the
primary deliverable (with full EDA, markdown explanations, and visuals),
but this script lets the same pipeline be re-run headlessly, e.g. for
regenerating best_model.pkl after a code change, without re-running the
whole notebook.

Usage:
    python src/train_model.py --data data/raw/flights.csv --sample-size 1000000
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import (
    drop_cancelled_and_diverted,
    drop_missing_target_rows,
    drop_duplicate_rows,
    validate_scheduled_departure,
    sample_dataset,
)
from feature_engineering import (
    engineer_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
)

RANDOM_STATE = 42


USE_COLS = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE", "SCHEDULED_TIME", "DISTANCE",
    "ARRIVAL_DELAY", "CANCELLED", "DIVERTED",
]

DTYPES = {
    "YEAR": "int16",
    "MONTH": "int8",
    "DAY": "int8",
    "DAY_OF_WEEK": "int8",
    "AIRLINE": "category",
    "ORIGIN_AIRPORT": "category",
    "DESTINATION_AIRPORT": "category",
    "SCHEDULED_DEPARTURE": "int16",
    "SCHEDULED_TIME": "float32",
    "DISTANCE": "float32",
    "ARRIVAL_DELAY": "float32",
    "CANCELLED": "int8",
    "DIVERTED": "int8",
}


def load_and_clean(path: str) -> pd.DataFrame:
    print(f"Loading dataset from {path} ...")
    # Only load the columns actually used, with memory-efficient dtypes.
    # Loading all 31 raw columns at default dtypes can exceed available RAM
    # on constrained machines.
    df = pd.read_csv(path, usecols=USE_COLS, dtype=DTYPES)
    print(f"Loaded shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    df = drop_duplicate_rows(df)
    df = drop_cancelled_and_diverted(df)
    df = drop_missing_target_rows(df, target_source_col="ARRIVAL_DELAY")
    df = validate_scheduled_departure(df)
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Delayed"] = (df["ARRIVAL_DELAY"] >= 15).astype(int)
    print("Target distribution:")
    print(df["Delayed"].value_counts(normalize=True).rename("proportion"))
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_transformer = StandardScaler()
    # sparse_output=False: HistGradientBoostingClassifier requires dense input.
    # max_categories=30: airports have ~628 unique values each; one-hot encoding
    # all of them creates 1000+ columns, which becomes too large as a dense
    # array. Capping to the most frequent 30 per column keeps memory bounded;
    # rare categories collapse into a single "infrequent" bucket.
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, max_categories=30, dtype=np.float32
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    print(f"\n{name} results:")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")
    return metrics


def main(data_path: str, sample_size: int, output_model_path: str, output_results_path: str):
    df = load_and_clean(data_path)
    df = build_target(df)

    if sample_size and len(df) > sample_size:
        df = sample_dataset(df, n=sample_size, target_col="Delayed", random_state=RANDOM_STATE)

    df = engineer_features(df)

    X = df[ALL_FEATURES]
    y = df["Delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = build_preprocessing_pipeline()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        print(f"\nTraining {name} ...")
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        results.append(evaluate(pipe, X_test, y_test, name))

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print("\nModel comparison:")
    print(results_df)
    results_df.to_csv(output_results_path, index=False)

    best_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model by F1-score: {best_name}")

    # Lightweight hyperparameter tuning example for the best model.
    if best_name == "Random Forest":
        param_dist = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_leaf": [1, 2, 5],
        }
        search = RandomizedSearchCV(
            best_pipeline, param_distributions=param_dist,
            cv=3, scoring="f1", random_state=RANDOM_STATE, n_iter=10, n_jobs=2
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        print(f"Best params: {search.best_params_}")
        evaluate(best_pipeline, X_test, y_test, f"{best_name} (tuned)")

    joblib.dump(best_pipeline, output_model_path)
    print(f"\nSaved final pipeline to {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/flights.csv")
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--output-model", default="models/best_model.pkl")
    parser.add_argument("--output-results", default="results/model_comparison.csv")
    args = parser.parse_args()

    main(args.data, args.sample_size, args.output_model, args.output_results)