"""负责人 A：数据获取检查、统一划分与输入准备。

任务：保留稳定 row_id，按 Time 顺序划分约 60/20/20，同时间边界不拆分。
输入：原始 creditcard.csv；输出：带 row_id 的数据表及三个划分。
row_id、Class 不作为模型特征。缩放器由各模型流程仅在训练数据上拟合。
状态：加载/检查/特征提取可用；划分尚未实现。
"""

from pathlib import Path


FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
ID_COLUMN = "row_id"


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
    if ID_COLUMN in data.columns:
        raise ValueError("Expected the original CSV without a row_id column.")
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[TARGET_COLUMN].isna().any() or not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1, with no missing labels.")
    if np.isinf(data[required].to_numpy(dtype=float)).any():
        raise ValueError("Numeric fields contain infinite values.")
    data.insert(0, ID_COLUMN, np.arange(len(data), dtype=np.int64))
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
        "exact_duplicate_rows": int(data[FEATURE_COLUMNS + [TARGET_COLUMN]].duplicated().sum()),
        "time_min": float(data["Time"].min()),
        "time_max": float(data["Time"].max()),
    }


def get_features(data):
    """Return only the declared inputs, excluding row_id, Class and extra columns."""
    return data.loc[:, FEATURE_COLUMNS].copy()


def split_data(data, fractions=(0.60, 0.20, 0.20)):
    """TODO: chronological split, keeping equal-time boundary records together.

    Return (train, validation, test) frames, each retaining row_id, Time, Class
    and the declared features. Check nonempty partitions and report class counts.
    Preserve row identity through sorting; never regenerate IDs per partition.
    Fit scaling, feature selection and resampling only within training workflows.
    """
    raise NotImplementedError("Implement the shared chronological split first.")

