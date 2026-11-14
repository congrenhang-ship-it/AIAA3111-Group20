"""负责人 A：相对时间窗口评价，建议投入 15%。

任务：按测试前确定的窗口分析固定模型，复用 C 的评价函数。
输入：evaluation.py 约定的预测表；输出：各窗口样本/欺诈数及审核指标。
窗口以原始 Time 的共同相对起点划分，所有方法使用同样窗口和预算。
这不是重新训练、调参或构造个人历史；两天数据不支持长期季节性结论。
状态：窗口划分与计数可用；指标委托 C 的公共评价。
"""

from math import isfinite

from evaluation import PREDICTION_COLUMNS


def _time_windows(predictions, window_seconds, origin):
    """Yield metadata and rows for each method and window in the observed span."""
    import numpy as np

    if not isfinite(window_seconds) or window_seconds <= 0 or not isfinite(origin):
        raise ValueError("Use a finite origin and a finite, positive window size.")
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if predictions.empty:
        raise ValueError("At least one prediction is needed to define the window span.")
    if predictions[["method", "row_id"]].isna().any().any():
        raise ValueError("Prediction method and row_id must be nonmissing.")
    if predictions.duplicated(["method", "row_id"]).any():
        raise ValueError("Each method must have one prediction per row_id.")
    if not predictions["Class"].isin([0, 1]).all():
        raise ValueError("Prediction labels must be 0 or 1.")
    times = predictions["Time"].to_numpy(dtype=float)
    if not np.isfinite(times).all():
        raise ValueError("Prediction times must be finite.")

    windows = np.floor((times - origin) / window_seconds)
    first_window, last_window = int(windows.min()), int(windows.max())
    table = predictions.assign(window_id=windows.astype("int64"))
    reference = None
    for method, method_rows in table.groupby("method", sort=False, observed=True):
        metadata = method_rows.set_index("row_id")[["Time", "Class"]].sort_index()
        if reference is None:
            reference = metadata
        elif not metadata.equals(reference):
            raise ValueError("All methods must use the same row_id, Time and Class records.")

        by_window = {key: rows for key, rows in method_rows.groupby("window_id", sort=False)}
        for window_id in range(first_window, last_window + 1):
            rows = by_window.get(window_id)
            n_samples = 0 if rows is None else len(rows)
            n_positives = 0 if rows is None else int(rows["Class"].sum())
            status = "ok"
            if rows is None:
                status = "empty"
            elif n_positives == 0:
                status = "no_positives"
            yield {
                "method": method,
                "window_id": window_id,
                "start_seconds": origin + window_id * window_seconds,
                "end_seconds": origin + (window_id + 1) * window_seconds,
                "n_samples": n_samples,
                "n_positives": n_positives,
                "status": status,
            }, rows


def summarize_time_windows(predictions, window_seconds=3600, origin=0):
    """Return window counts without invoking the public evaluator."""
    return [summary for summary, _ in _time_windows(predictions, window_seconds, origin)]


def evaluate_time_windows(
    predictions, window_seconds=3600, origin=0,
    review_fractions=(0.001, 0.005, 0.01, 0.02), threshold=None,
):
    """Return one row per method/window with C's metrics under 'evaluation'.

    Use independent budget allocations in each nonempty window. Empty windows
    have no evaluation. C supplies undefined AP and recall for no-positive windows.
    """
    from evaluation import evaluate_scores

    results = []
    for summary, rows in _time_windows(predictions, window_seconds, origin):
        summary["evaluation"] = None
        if rows is not None:
            summary["evaluation"] = evaluate_scores(
                rows["Class"], rows["score"], review_fractions,
                row_ids=rows["row_id"], threshold=threshold,
            )
        results.append(summary)
    return results


def select_review_threshold(predictions, review_fraction):
    """Select the Kth score once on validation and record its boundary ties."""
    from math import ceil
    import numpy as np

    if predictions.empty or predictions["method"].nunique(dropna=False) != 1:
        raise ValueError("Threshold selection requires one nonempty validation method.")
    if not isfinite(review_fraction) or not 0 < review_fraction <= 1:
        raise ValueError("The review fraction must be in (0, 1].")
    scores = predictions["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires finite scores.")
    k = ceil(len(scores) * review_fraction)
    threshold = float(np.partition(scores, len(scores) - k)[len(scores) - k])
    return {
        "selection_phase": "validation",
        "review_fraction": review_fraction,
        "target_K": k,
        "threshold": threshold,
        "strictly_above_threshold": int((scores > threshold).sum()),
        "equal_to_threshold": int((scores == threshold).sum()),
        "decision_rule": "score >= threshold",
    }


def evaluate_review_threshold(predictions, selection, *, window_seconds=3600, origin=0):
    """Apply an existing threshold overall and in fixed windows without tuning it."""
    from evaluation import evaluate_scores

    q = selection["review_fraction"]
    threshold = selection["threshold"]

    def counts(metrics):
        n = metrics["n_samples"]
        fixed = metrics["threshold"]
        return {
            "fixed_threshold": {**fixed, "actual_review_fraction": fixed["K"] / n if n else None},
            "top_k": metrics["budgets"][0],
        }

    windows = evaluate_time_windows(predictions, window_seconds, origin, (q,), threshold)
    for window in windows:
        metrics = window.pop("evaluation")
        if metrics is None:
            rows = predictions.iloc[:0]
            metrics = evaluate_scores(rows.Class, rows.score, (q,), row_ids=rows.row_id, threshold=threshold)
        window.update(counts(metrics))
    overall = evaluate_scores(
        predictions.Class, predictions.score, (q,), row_ids=predictions.row_id, threshold=threshold,
    )
    return {"overall": counts(overall), "windows": windows}
