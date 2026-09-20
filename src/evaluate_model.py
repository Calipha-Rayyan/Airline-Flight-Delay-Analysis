"""
evaluate_model.py

Loads a saved pipeline (preprocessing + model) and produces the required
evaluation visualizations: confusion matrix, ROC curve, and feature
importance (for tree-based models). Run this after train_model.py, or
call these functions from the notebook.

Usage:
    python src/evaluate_model.py --model models/best_model.pkl \
        --data data/raw/flights.csv --sample-size 1000000
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from data_cleaning import (
    drop_cancelled_and_diverted,
    drop_missing_target_rows,
    drop_duplicate_rows,
    validate_scheduled_departure,
    sample_dataset,
)
from feature_engineering import engineer_features, ALL_FEATURES, CATEGORICAL_FEATURES

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


def prepare_holdout(data_path: str, sample_size: int):
    df = pd.read_csv(data_path, usecols=USE_COLS, dtype=DTYPES)
    df = drop_duplicate_rows(df)
    df = drop_cancelled_and_diverted(df)
    df = drop_missing_target_rows(df, target_source_col="ARRIVAL_DELAY")
    df = validate_scheduled_departure(df)
    df["Delayed"] = (df["ARRIVAL_DELAY"] >= 15).astype(int)

    if sample_size and len(df) > sample_size:
        df = sample_dataset(df, n=sample_size, target_col="Delayed", random_state=RANDOM_STATE)

    df = engineer_features(df)
    X = df[ALL_FEATURES]
    y = df["Delayed"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    return X_test, y_test


def plot_confusion_matrix(pipeline, X_test, y_test, output_path: str):
    preds = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Delayed", "Delayed"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — Final Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved confusion matrix to {output_path}")
    plt.close(fig)


def plot_roc_curve(pipeline, X_test, y_test, output_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve — Final Model", fontsize=13)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved ROC curve to {output_path}")
    plt.close(fig)


def plot_feature_importance(pipeline, output_path: str, top_n: int = 20):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        print(f"Model {type(model).__name__} has no feature_importances_; skipping.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = list(num_features) + list(cat_feature_names)

    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — Final Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved feature importance chart to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--data", default="data/raw/flights.csv")
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    X_test, y_test = prepare_holdout(args.data, args.sample_size)

    plot_confusion_matrix(pipeline, X_test, y_test, f"{args.results_dir}/confusion_matrix.png")
    plot_roc_curve(pipeline, X_test, y_test, f"{args.results_dir}/roc_curve.png")
    plot_feature_importance(pipeline, f"{args.results_dir}/feature_importance.png")