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

DAY_SECONDS = 86400


def handbook_history_builder(context, *, history_keep=None):
    """Create four row-keyed views using only visible events at strictly earlier times.

    Windows are [t - window, t). history_keep controls history membership only.
    Querying all rows with searchsorted(left) enforces batch isolation on ties.
    """
    import numpy as np
    import pandas as pd

    columns = {"row_id", "Time", "CUSTOMER_ID", "TERMINAL_ID", "Amount"}
    if set(context.columns) != columns or not context.columns.is_unique or context.empty:
        raise ValueError("Provide nonempty, unlabeled Handbook context with the five declared columns")
    if context.isna().any().any() or not context.row_id.is_unique:
        raise ValueError("Context fields must be present and row_id unique")
    for field in ("row_id", "CUSTOMER_ID", "TERMINAL_ID"):
        if not pd.api.types.is_integer_dtype(context[field]) or (context[field] < 0).any():
            raise ValueError(f"{field} must be a nonnegative integer")
    index = pd.Index(context.row_id, name="row_id")
    if history_keep is None:
        keep = np.ones(len(context), dtype=bool)
    else:
        if (not isinstance(history_keep, pd.Series) or not pd.api.types.is_bool_dtype(history_keep)
                or not history_keep.index.is_unique or len(history_keep) != len(index)
                or history_keep.isna().any() or not index.isin(history_keep.index).all()):
            raise ValueError("history_keep must be a boolean Series covering every source row_id")
        keep = history_keep.reindex(index).to_numpy(dtype=bool)
    amount = context.Amount.to_numpy(dtype=float)
    scaled_amount = amount * 100
    rounded = np.rint(scaled_amount)
    if (not np.isfinite(amount).all() or (amount < 0).any()
            or not np.allclose(scaled_amount, rounded, rtol=0, atol=1e-7)
            or rounded.max() >= np.iinfo(np.int64).max // len(context)):
        raise ValueError("Handbook Amount must be finite, nonnegative and safely expressible in cents")
    cents = rounded.astype(np.int64)
    # The shared prediction/relationship APIs normalize Time to floating point.
    seconds = context.Time.to_numpy(dtype=float)
    if (not np.isfinite(seconds).all() or (seconds < 0).any()
            or (seconds >= 2**53).any() or not np.equal(seconds, np.floor(seconds)).all()):
        raise ValueError("Time must contain nonnegative whole seconds")
    times = seconds.astype(np.int64)
    angle = 2 * np.pi * (times % DAY_SECONDS) / DAY_SECONDS
    single = pd.DataFrame({"Amount": amount, "time_sin": np.sin(angle), "time_cos": np.cos(angle)}, index=index)
    histories = {}
    for entity in ("customer", "terminal"):
        entities = context[f"{entity.upper()}_ID"].to_numpy()
        order = np.lexsort((times, entities))
        boundaries = np.flatnonzero(entities[order][1:] != entities[order][:-1]) + 1
        counts = np.zeros((2, len(context)), dtype=np.int64)
        sums = np.zeros_like(counts)
        for positions in np.split(order, boundaries):
            queries = times[positions]
            retained = positions[keep[positions]]
            visible_times = times[retained]
            prefix = np.r_[0, np.cumsum(cents[retained])]
            right = np.searchsorted(visible_times, queries, side="left")
            for i, days in enumerate((1, 7)):
                left = np.searchsorted(visible_times, queries - days * DAY_SECONDS, side="left")
                counts[i, positions] = right - left
                sums[i, positions] = prefix[right] - prefix[left]
        values = {}
        for i, days in enumerate((1, 7)):
            count, total = counts[i], sums[i]
            period = f"{days}d"
            values[f"{entity}_count_{period}"] = count
            values[f"{entity}_mean_amount_{period}"] = np.divide(
                total, 100 * count, out=np.zeros(len(context)), where=count != 0
            )
            values[f"{entity}_no_history_{period}"] = (count == 0).astype(np.int8)
            if entity == "customer":
                values[f"customer_amount_ratio_{period}"] = np.divide(
                    cents * count.astype(float), total, out=np.zeros(len(context)), where=total > 0
                )
                values[f"customer_amount_ratio_unavailable_{period}"] = (total <= 0).astype(np.int8)
        histories[entity] = pd.DataFrame(values, index=index)
    customer, terminal = histories["customer"], histories["terminal"]
    return {
        "transaction_only": single,
        "customer_history": pd.concat([single, customer], axis=1),
        "terminal_history": pd.concat([single, terminal], axis=1),
        "both_history": pd.concat([single, customer, terminal], axis=1),
    }

def run_handbook_rolling(development, output):
    """Fit E11 on legal development labels and save paired model/threshold snapshots."""
    import hashlib
    import tempfile
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd

    from classification import score_model, train_model
    from evaluation import evaluate_amounts, fit_amount_bins, make_prediction_frame, partition_description
    from handbook import write_experiment_json
    from preprocessing import data_sha256

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    train, validation = (development[key] for key in ("train", "validation"))
    manifest = development["manifest"]
    train_end, test_start = (manifest["partition_days"][key][0] for key in ("validation", "test"))
    all_rows = pd.concat([train, validation])
    if (train.Time.max() >= train_end * DAY_SECONDS or validation.Time.min() < train_end * DAY_SECONDS
            or validation.Time.max() >= test_start * DAY_SECONDS):
        raise ValueError("E11 requires only the declared chronological development rows")
    features = pd.concat([development[f"{key}_views"]["both_history"] for key in ("train", "validation")])
    boundaries = [*range(train_end, test_start, 7), test_start]
    evaluation_sets = []
    for first, last in zip(boundaries, boundaries[1:]):
        window = validation.loc[validation.Time.ge(first * DAY_SECONDS) & validation.Time.lt(last * DAY_SECONDS)]
        if window.empty:
            raise ValueError(f"Validation interval [{first}, {last}) has no transactions")
        evaluation_sets.append((first, last, window))
    bins = fit_amount_bins(train)
    specifications = {"static": None, "weekly_delay_0d": 0, "weekly_delay_1d": 1, "weekly_delay_7d": 7}
    summary = {
        "dataset_id": "handbook", "experiment": "E11", "phase": "validation",
        "data_sha256": manifest["data_sha256"], "development_partitions": {k: manifest["partitions"][k] for k in ("train", "validation")},
        "classifier": {"model": "logistic_regression", "strategy": "class_weight", "seed": 42},
        "feature_view": "both_history", "feature_columns": list(features.columns),
        "window_days": 30, "update_every_days": 7, "scheduled_update_days": boundaries[:-1],
        "time_origin_day": 0, "validation_days": [train_end, test_start],
        "amount_bins": bins,
        "threshold_calibration": "Each final saved model scores the same untouched last validation window at 1%.",
        "test_scored": False, "group_G2_complete": False, "conditions": {},
    }

    def fit(rows, cutoff, delay):
        if not (rows.Time + delay * DAY_SECONDS < cutoff * DAY_SECONDS).all():
            raise ValueError("E11 fit contains unavailable labels")
        if set(rows.Class) != {0, 1}:
            raise ValueError(f"E11 day {cutoff} / delay {delay} fit needs both classes")
        return train_model(features.loc[rows.row_id], rows.Class, "logistic_regression", "class_weight", 42)

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "e11"
        stage.mkdir()
        static = fit(train, train_end, 0)
        for method, delay in specifications.items():
            predictions, windows = [], []
            model = static
            for first, last, window in evaluation_sets:
                cutoff = train_end if delay is None else first
                observed_delay = 0 if delay is None else delay
                upper = cutoff - observed_delay
                lower = max(0, upper - 30)
                fit_rows = train if delay is None else all_rows.loc[
                    all_rows.Time.ge(lower * DAY_SECONDS) & all_rows.Time.lt(upper * DAY_SECONDS)
                ]
                if delay is not None:
                    model = fit(fit_rows, cutoff, observed_delay)
                prediction = make_prediction_frame(window, score_model(model, features.loc[window.row_id]), method)
                predictions.append(prediction)
                windows.append({
                    "score_days": [first, last], "fit_decision_day": cutoff,
                    "fit_transaction_days": [lower, upper], "fit": partition_description(fit_rows),
                    "latest_label_available_second": int(fit_rows.Time.max()) + observed_delay * DAY_SECONDS,
                    "validation": partition_description(window),
                    "metrics": evaluate_amounts(prediction, window, bins),
                })
            all_predictions = pd.concat(predictions)
            all_predictions.to_csv(stage / f"{method}_validation.csv", index=False, lineterminator="\n")
            bundle = stage / f"{method}.joblib"
            joblib.dump(model, bundle)
            calibration = predictions[-1]
            final_X = features.loc[calibration.row_id]
            if not np.array_equal(score_model(joblib.load(bundle), final_X), calibration.score.to_numpy()):
                raise RuntimeError("Saved E11 snapshot does not replay its calibration scores")
            snapshot = {
                "model_file": bundle.name, "model_sha256": data_sha256(bundle),
                "fit": windows[-1]["fit"], "fit_decision_day": windows[-1]["fit_decision_day"],
                "validation": windows[-1]["validation"],
                "calibration_scores_sha256": hashlib.sha256(np.asarray(calibration.score, dtype="<f8").tobytes()).hexdigest(),
                "threshold": select_review_threshold(calibration, 0.01),
            }
            threshold_file = stage / f"{method}_threshold.json"
            write_experiment_json(threshold_file, snapshot)
            summary["conditions"][method] = {
                "delay_days": delay, "fits": 1 if delay is None else len(windows), "windows": windows,
                "pooled_prequential_metrics": evaluate_amounts(all_predictions, validation, bins),
                "snapshot": {**snapshot, "threshold_file": threshold_file.name, "threshold_sha256": data_sha256(threshold_file)},
            }
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 5), layout="constrained")
        axes = figure.subplots(2, 1, sharex=True)
        for method, bank in summary["conditions"].items():
            x = [window["score_days"][0] for window in bank["windows"]]
            axes[0].plot(x, [window["metrics"]["AP"] for window in bank["windows"]], marker="o", label=method)
            recall = [next(row["recall_at_k"] for row in window["metrics"]["budgets"] if row["review_fraction"] == 0.01)
                      for window in bank["windows"]]
            axes[1].plot(x, recall, marker="o")
        axes[0].set(ylabel="Average precision", title="Handbook E11 development validation")
        axes[0].legend(ncols=2)
        axes[1].set(ylabel="Recall at 1%", xlabel="Validation window start (day)", xticks=boundaries[:-1])
        figure.savefig(stage / "rolling_validation.png", dpi=160)
        write_experiment_json(stage / "summary.json", summary)
        stage.rename(output)
    return summary
