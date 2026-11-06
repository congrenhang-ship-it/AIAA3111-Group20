"""Data loading and audit; splitting and preprocessing are not yet implemented."""

from pathlib import Path


FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"


def load_data(path: Path):
    """Load the original CSV and check its required numeric fields."""
    import numpy as np
    import pandas as pd

    data = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("The data file contains no transactions.")
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[TARGET_COLUMN].isna().any() or not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1, with no missing labels.")
    if np.isinf(data[required].to_numpy(dtype=float)).any():
        raise ValueError("Numeric fields contain infinite values.")
    return data


def audit_data(data):
    """Describe data without deleting, scaling or resampling any rows."""
    labels = data[TARGET_COLUMN]
    return {
        "rows": len(data),
        "columns": list(data.columns),
        "class_counts": {str(int(k)): int(v) for k, v in labels.value_counts().items()},
        "fraud_fraction": float(labels.mean()),
        "missing_values": {str(k): int(v) for k, v in data.isna().sum().items()},
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "time_min": float(data["Time"].min()),
        "time_max": float(data["Time"].max()),
    }


def split_data(data, fractions=(0.60, 0.20, 0.20)):
    """TODO: chronological split, keeping equal-time boundary records together.

    Return train, validation and test frames. Record class counts and preserve
    row identity. Fit scaling, feature selection and resampling only afterward
    within the training workflow.
    """
    raise NotImplementedError("Implement the shared chronological split first.")

