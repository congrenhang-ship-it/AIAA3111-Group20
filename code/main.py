"""负责人 A：运行入口与流程整合。

任务：统一数据路径、随机种子、时间划分和审核预算；接入 B/C/D 的模块。
路线：数据 → 分类/聚类/异常检测/规则 → 验证冻结 → 测试 → 时间分析与图表。
当前可用：数据审计、划分检查、G1 验证入口。G1 指标依赖 B2/C2。
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from preprocessing import (
    DEFAULT_DATA_PATH,
    ID_COLUMN,
    SPLIT_FRACTIONS,
    TARGET_COLUMN,
    ULB_SHA256,
    audit_data,
    data_sha256,
    get_features,
    load_data,
    split_data,
)


SEED = 42
REVIEW_FRACTIONS = (0.001, 0.005, 0.01, 0.02)
TIME_WINDOW_SECONDS = 3600  # 初始建议；窗口口径须在最终测试前确定。

TASKS = {
    "A": ["main.py: 流程整合", "preprocessing.py: 数据与时间划分", "temporal.py: 时间窗口分析"],
    "B": ["classification.py: 分类模型、不平衡处理、风险分数"],
    "C": ["unsupervised.py: K-means、Isolation Forest、簇距离", "evaluation.py: 评价、等预算组合、图表"],
    "D": ["pattern_mining.py: 分箱、Apriori、规则验证", "report/、poster/: 文档整合，全组提供内容"],
}


def split_summary(parts):
    """Report split counts and a fingerprint of the ordered source row IDs."""
    summary = {}
    for name, data in zip(("train", "validation", "test"), parts, strict=True):
        fingerprint = sha256()
        for row_id in data[ID_COLUMN]:
            fingerprint.update(f"{row_id}\n".encode("ascii"))
        summary[name] = {
            "rows": len(data),
            "frauds": int(data[TARGET_COLUMN].sum()),
            "time_min": float(data["Time"].iloc[0]),
            "time_max": float(data["Time"].iloc[-1]),
            "row_ids_sha256": fingerprint.hexdigest(),
        }
    return summary


def run_validation(train, validation):
    """Run the first G1 classification baseline using B and C's interfaces."""
    import numpy as np

    from classification import score_model, train_model
    from evaluation import evaluate_scores, make_prediction_frame
    from temporal import evaluate_time_windows

    model_name = "logistic_regression"
    strategy = "original"
    train_features = get_features(train)
    try:
        model = train_model(
            train_features, train[TARGET_COLUMN], model_name, strategy=strategy, seed=SEED,
        )
        scores = score_model(model, get_features(validation))
    except NotImplementedError as exc:
        raise RuntimeError(f"G1 is waiting for B2 (classification): {exc}") from exc

    try:
        predictions = make_prediction_frame(validation, scores, model_name)
        metadata = [ID_COLUMN, TARGET_COLUMN, "Time"]
        if not predictions[metadata].reset_index(drop=True).equals(
            validation[metadata].reset_index(drop=True)
        ):
            raise ValueError("C's prediction frame must preserve validation rows and their order.")
        if not predictions["method"].isin([model_name]).all():
            raise ValueError("C's prediction frame must preserve the requested method name.")
        if not np.isfinite(predictions["score"].to_numpy(dtype=float)).all():
            raise ValueError("C's prediction frame must contain finite risk scores.")
        overall = evaluate_scores(
            predictions[TARGET_COLUMN], predictions["score"], REVIEW_FRACTIONS,
            row_ids=predictions[ID_COLUMN],
        )
        windows = evaluate_time_windows(
            predictions, window_seconds=TIME_WINDOW_SECONDS, origin=0,
            review_fractions=REVIEW_FRACTIONS,
        )
    except NotImplementedError as exc:
        raise RuntimeError(f"G1 is waiting for C2 (shared evaluation): {exc}") from exc

    return {
        "split": "validation",
        "model": model_name,
        "strategy": strategy,
        "seed": SEED,
        "feature_columns": list(train_features.columns),
        "review_fractions": REVIEW_FRACTIONS,
        "window_seconds": TIME_WINDOW_SECONDS,
        "window_origin": 0,
        "overall": overall,
        "time_windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Group 20 data preparation and validation baseline."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH,
        help="Path to the original creditcard.csv file.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-data", action="store_true", help="Run a read-only data audit.")
    action.add_argument("--prepare-data", action="store_true", help="Verify ULB and report the shared splits.")
    action.add_argument("--run-validation", action="store_true", help="Run the G1 baseline on validation only.")
    parser.add_argument("--tasks", action="store_true", help="Show agreed module ownership.")
    args = parser.parse_args()
    run_data_action = args.check_data or args.prepare_data or args.run_validation
    if args.tasks:
        print(json.dumps(TASKS, indent=2, ensure_ascii=False))
    if not run_data_action:
        if not args.tasks:
            parser.print_help()
        return 0
    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}. See data/README.md.")
    try:
        sha256 = data_sha256(args.data)
        matches_ulb = sha256 == ULB_SHA256
        if not args.check_data and not matches_ulb:
            raise ValueError(f"ULB SHA-256 mismatch: {sha256}. See data/README.md.")
        data = load_data(args.data)
        if args.check_data:
            summary = audit_data(data)
        else:
            parts = split_data(data)
            summary = {
                "split_fractions_target": SPLIT_FRACTIONS,
                "rows_removed_as_exact_duplicates": len(data) - sum(map(len, parts)),
                "partitions": split_summary(parts),
            }
            if args.run_validation:
                summary["experiment"] = run_validation(parts[0], parts[1])
        summary.update(sha256=sha256, matches_ulb_v1=matches_ulb)
        output = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    except ModuleNotFoundError:
        parser.error("Install dependencies with: python -m pip install -r requirements.txt")
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

