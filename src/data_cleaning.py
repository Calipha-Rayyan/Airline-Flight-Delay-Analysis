"""
data_cleaning.py

Reusable cleaning functions for the flights dataset. Each function documents
WHY a cleaning decision is made, in line with the project's "no silent
drops" rule. Import and call these from the notebook so cleaning logic is
version-controlled and testable rather than living only in notebook cells.
"""

import pandas as pd
import numpy as np


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy summary of missing values per column, sorted descending."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def drop_cancelled_and_diverted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cancelled flights have no ARRIVAL_DELAY (the flight never happened), so
    they cannot be labeled Delayed/Not-Delayed and must be excluded from
    a model that predicts arrival delay.

    Diverted flights have unreliable/incomplete arrival timing and are
    excluded for the same reason: the target would not be well-defined.

    This is an explicit, documented drop -- not silent.
    """
    before = len(df)
    out = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
    after = len(out)
    print(f"Dropped {before - after:,} cancelled/diverted rows "
          f"({(before - after) / before * 100:.2f}% of data). "
          "Reason: no valid ARRIVAL_DELAY target for these flights.")
    return out


def drop_missing_target_rows(df: pd.DataFrame, target_source_col: str = "ARRIVAL_DELAY") -> pd.DataFrame:
    """
    Rows with a missing ARRIVAL_DELAY cannot have a target label computed.
    These are dropped explicitly, with the count reported.
    """
    before = len(df)
    out = df.dropna(subset=[target_source_col]).copy()
    after = len(out)
    print(f"Dropped {before - after:,} rows with missing {target_source_col} "
          f"({(before - after) / before * 100:.2f}% of data). "
          "Reason: cannot compute target label without this value.")
    return out


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, reporting how many were found."""
    before = len(df)
    out = df.drop_duplicates().copy()
    after = len(out)
    print(f"Dropped {before - after:,} exact duplicate rows.")
    return out


def validate_scheduled_departure(df: pd.DataFrame) -> pd.DataFrame:
    """
    SCHEDULED_DEPARTURE should be a valid HHMM value between 0 and 2359,
    with minute component 00-59. Rows failing this are invalid timestamps
    and are removed, since engineered time features cannot be trusted
    otherwise.
    """
    before = len(df)
    sd = df["SCHEDULED_DEPARTURE"].astype(int)
    valid_mask = (sd >= 0) & (sd <= 2359) & ((sd % 100) <= 59)
    out = df[valid_mask].copy()
    after = len(out)
    print(f"Dropped {before - after:,} rows with invalid SCHEDULED_DEPARTURE "
          "timestamps (outside valid HHMM range).")
    return out


def sample_dataset(df: pd.DataFrame, n: int, target_col: str, random_state: int = 42) -> pd.DataFrame:
    """
    Take a reproducible stratified sample of the dataset for modeling,
    preserving the class distribution of target_col.

    Documents: original size, sample size, method, and random_state, per
    project reporting requirements.

    Implementation note: this builds the sample by looping over each class
    and sampling within it, then concatenating and shuffling. This avoids
    groupby(...).apply(...), whose behavior around the grouping column
    changed across pandas versions and can silently drop target_col.
    """
    original_size = len(df)
    if original_size <= n:
        print(f"Dataset size ({original_size:,}) already <= requested sample "
              f"size ({n:,}); using full dataset, no sampling applied.")
        return df.copy()

    frac = n / original_size

    parts = []
    for class_value, group in df.groupby(target_col):
        parts.append(group.sample(frac=frac, random_state=random_state))

    sampled = pd.concat(parts, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print("Sampling summary:")
    print(f"  Original size:   {original_size:,}")
    print(f"  Sample size:     {len(sampled):,}")
    print(f"  Method:          stratified by '{target_col}', frac={frac:.4f}")
    print(f"  Random state:    {random_state}")
    print("  Reason:          full dataset too large for practical model "
          "training/tuning on constrained hardware; sampling preserves class balance.")
    return sampled