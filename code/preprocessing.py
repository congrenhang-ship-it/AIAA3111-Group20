"""负责人 A：数据获取检查、统一划分与输入准备。

任务：保留稳定 row_id，按 Time 顺序划分约 60/20/20，同时间边界不拆分。
输入：原始 creditcard.csv；输出：带 row_id 的数据表及三个划分。
row_id、Class 不作为模型特征。缩放器由各模型流程仅在训练数据上拟合。
状态：加载、审计、去重、时间划分和特征提取可用。
"""

from hashlib import file_digest
from math import isclose, isfinite
from pathlib import Path


FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
ID_COLUMN = "row_id"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"
ULB_DOWNLOAD_URL = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"
ULB_SIZE_BYTES = 150_828_752
ULB_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)


def data_sha256(path: Path):
    """Return the SHA-256 of the original CSV bytes."""
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


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
    duplicates = data.loc[data.duplicated(FEATURE_COLUMNS + [TARGET_COLUMN]), TARGET_COLUMN]
    summary = {
        "rows": len(data),
        "columns": list(data.columns),
        "class_counts": {"0": int(labels.eq(0).sum()), "1": int(labels.eq(1).sum())},
        "fraud_fraction": float(labels.mean()),
        "missing_values": {str(k): int(v) for k, v in data.isna().sum().items()},
        "exact_duplicate_rows": len(duplicates),
        "duplicate_class_counts": {
            "0": int(duplicates.eq(0).sum()),
            "1": int(duplicates.eq(1).sum()),
        },
    }
    for column in ("Time", "Amount"):
        values = data[column].dropna()
        summary[f"{column.lower()}_min"] = None if values.empty else float(values.min())
        summary[f"{column.lower()}_max"] = None if values.empty else float(values.max())
    return summary


def get_features(data):
    """Return only the declared inputs, excluding row_id, Class and extra columns."""
    return data.loc[:, FEATURE_COLUMNS].copy()


def split_data(data, fractions=SPLIT_FRACTIONS):
    """Deduplicate and return (train, validation, test) in time order.

    Keep the first source row of each exact 31-field duplicate. Choose cut
    timestamps by row position after cleaning and put all ties on the later side.
    Preserve original row_id values and leave feature fitting to the callers.
    """
    if (
        len(fractions) != 3
        or any(not isfinite(value) or value <= 0 for value in fractions)
        or not isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("Expected three positive split fractions summing to 1.")
    if data[ID_COLUMN].isna().any() or not data[ID_COLUMN].is_unique:
        raise ValueError("row_id must be nonmissing and unique before splitting.")
    missing = data[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum()
    if missing.any():
        raise ValueError(f"Missing required values; audit before splitting: {missing[missing > 0].to_dict()}")

    ordered = (
        data.sort_values(ID_COLUMN)
        .drop_duplicates(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        .sort_values(["Time", ID_COLUMN])
    )
    if len(ordered) < 3:
        raise ValueError("At least three distinct transactions are needed for a time split.")
    time = ordered["Time"]
    train_target = int(len(ordered) * fractions[0])
    test_target = int(len(ordered) * sum(fractions[:2]))
    if not 0 < train_target < test_target < len(ordered):
        raise ValueError("The split fractions do not allocate rows to all three partitions.")
    train_end = time.searchsorted(time.iloc[train_target], side="left")
    validation_end = time.searchsorted(time.iloc[test_target], side="left")
    if not 0 < train_end < validation_end < len(ordered):
        raise ValueError("The proposed cuts do not yield three nonempty time partitions.")

    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )

