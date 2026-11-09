"""负责人 D：训练分箱、Apriori 短规则、留出覆盖验证。

输入：A 的划分表（稳定整数 row_id、二元 Class、少量数值特征）。
输出：mine_rules -> (可 JSON 序列化的训练分箱信息, 规则列表)；
validate_rules -> 同顺序的留出指标与覆盖 row_id，不修改规则或分箱。
状态：合成事务开发；尚未在 A 的统一真实划分上运行或冻结方案。
规则不是因果解释，不赋予匿名特征未知的业务含义。
"""

from itertools import combinations
from numbers import Integral

import numpy as np
import pandas as pd


# Proposed training-only limits; real-data validation and G2 freeze are separate.
MIN_COVERAGE = 20
MIN_FRAUD_COVERAGE = 5
MAX_FEATURES = 5
MAX_RULES = 50


def _check_data(data, features):
    missing = sorted(set(["row_id", "Class", *features]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    ids = data["row_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("row_id must be nonmissing and unique within the split.")
    if not all(isinstance(value, Integral) and value >= 0 for value in ids):
        raise ValueError("row_id must retain nonnegative integer CSV row identities.")
    if not data["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1, without missing labels.")
    for feature in features:
        if not pd.api.types.is_numeric_dtype(data[feature]):
            raise ValueError(f"{feature} must be numeric.")
        if not np.isfinite(data[feature].to_numpy(dtype=float)).all():
            raise ValueError(f"{feature} contains missing or nonfinite values.")


def _encode(data, fitted_binner):
    features = fitted_binner["feature_columns"]
    _check_data(data, features)
    encoded = {}
    for feature in features:
        cuts = np.asarray(fitted_binner["features"][feature]["cut_points"], dtype=float)
        if not np.isfinite(cuts).all() or (np.diff(cuts) <= 0).any():
            raise ValueError("Stored cut points must be finite and strictly increasing.")
        # [lower, upper): equality goes right; outer bins extend to infinity.
        encoded[feature] = np.searchsorted(
            cuts, data[feature].to_numpy(dtype=float), side="right"
        )
    return encoded


def _metrics(data, mask):
    labels = data["Class"].to_numpy(dtype=int)
    n, positives, coverage = len(data), int(labels.sum()), int(mask.sum())
    fraud = int(labels[mask].sum())
    confidence = fraud / coverage if coverage else None
    recall = fraud / positives if positives else None
    prevalence = positives / n if n else None
    return {
        "n_transactions": n, "n_fraud": positives,
        "coverage_count": coverage, "normal_count": coverage - fraud,
        "fraud_count": fraud,
        "antecedent_support": coverage / n if n else None,
        "support": fraud / n if n else None,
        "confidence": confidence,
        "lift": confidence / prevalence if coverage and positives else None,
        "fraud_recall": recall,
        "kulczynski": (confidence + recall) / 2 if coverage and positives else None,
        "covered_row_ids": [int(value) for value in data.loc[mask, "row_id"]],
        "status": "empty" if not n else "no_fraud" if not positives else "ok",
    }


def mine_rules(train_data, feature_columns, bins=3, max_length=3, binning="equal_frequency"):
    """Fit training bins and mine feature antecedents -> Class=1 using Apriori.

    Public signature unchanged. Return (binner_dict, rules). max_length includes
    the one-item consequent (2..3); search uses <=5 features and 2..5 bins.
    Explicit feature choices must come from training only. Require antecedent
    coverage >=20, fraud coverage >=5, and training lift >1. Keep at most 50
    rules ordered using training metrics only. These are development settings,
    not evidence of reliability or a frozen real-data experimental choice.
    """
    if isinstance(feature_columns, str):
        raise ValueError("feature_columns must be a sequence of feature names.")
    features = list(feature_columns)
    allowed = {"Time", "Amount", *[f"V{i}" for i in range(1, 29)]}
    if not 1 <= len(features) <= MAX_FEATURES or len(set(features)) != len(features):
        raise ValueError("Provide one to five distinct features.")
    if not set(features) <= allowed:
        raise ValueError("Use declared transaction features, never Class or row_id.")
    if isinstance(bins, bool) or not isinstance(bins, Integral) or not 2 <= bins <= 5:
        raise ValueError("bins must be an integer from 2 to 5.")
    if (isinstance(max_length, bool) or not isinstance(max_length, Integral)
            or not 2 <= max_length <= 3):
        raise ValueError("max_length must be 2 or 3, including the consequent.")
    if binning not in {"equal_width", "equal_frequency"}:
        raise ValueError("binning must be equal_width or equal_frequency.")
    _check_data(train_data, features)
    if train_data.empty:
        raise ValueError("Cannot fit bins on an empty training split.")
    binner = {
        "schema_version": 1, "feature_columns": features, "binning": binning,
        "requested_bins": int(bins), "max_length": int(max_length),
        "interval_policy": "left_closed_cut_goes_right_outer_bins_unbounded",
        "missing_policy": "raise",
        "limits": {"min_coverage": MIN_COVERAGE,
                   "min_fraud_coverage": MIN_FRAUD_COVERAGE,
                   "min_lift_exclusive": 1.0, "max_rules": MAX_RULES},
        "features": {},
    }
    for feature in features:
        values = train_data[feature].to_numpy(dtype=float)
        lo, hi = float(values.min()), float(values.max())
        raw = (np.linspace(lo, hi, bins + 1)[1:-1] if binning == "equal_width"
               else np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        # Collapse repeated quantiles/constant columns, without artificial jitter.
        cuts = np.unique(raw[(raw > lo) & (raw < hi)])
        binner["features"][feature] = {
            "cut_points": cuts.tolist(), "train_min": lo, "train_max": hi,
            "effective_bins": len(cuts) + 1,
        }
    encoded = _encode(train_data, binner)
    masks = {(feature, value): encoded[feature] == value
             for feature in features for value in sorted(set(encoded[feature]))}
    # Frequent feature-only itemsets; downward closure of coverage prunes
    # supersets safely. Class is never an antecedent or a binning input.
    level = {(item,): mask for item, mask in masks.items()
             if int(mask.sum()) >= MIN_COVERAGE}
    candidates = []
    prevalence = float(train_data["Class"].mean())
    for size in range(1, max_length):
        for items, mask in sorted(level.items()):
            fraud = int(train_data.loc[mask, "Class"].sum())
            if fraud >= MIN_FRAUD_COVERAGE and fraud / int(mask.sum()) > prevalence:
                candidates.append((items, mask, fraud / int(mask.sum()) / prevalence, fraud))
        if size == max_length - 1:
            break
        next_level = {}
        for left, right in combinations(sorted(level), 2):
            items = tuple(sorted(set(left) | set(right)))
            if len(items) != size + 1 or len({f for f, _ in items}) != len(items):
                continue
            if items in next_level or not all(tuple(sub) in level for sub in combinations(items, size)):
                continue
            mask = np.logical_and.reduce([masks[item] for item in items])
            if int(mask.sum()) >= MIN_COVERAGE:
                next_level[items] = mask
        level = next_level
    candidates.sort(key=lambda item: (-item[2], -item[3], len(item[0]), item[0]))
    rules = []
    for index, (items, mask, _, _) in enumerate(candidates[:MAX_RULES], start=1):
        rules.append({
            "rule_id": f"R{index:03d}",
            "antecedent": [{"feature": f, "bin": int(b)} for f, b in items],
            "consequent": {"Class": 1}, "train": _metrics(train_data, mask),
        })
    return binner, rules


def validate_rules(data, fitted_binner, rules):
    """Apply training bins/rules unchanged and return same-order metric dicts.

    First use on validation; A evaluates test only after G2 freezes rule IDs.
    Zero denominators -> None (JSON null), never invented infinite lift.
    support=P(A AND fraud), confidence=P(fraud|A), lift=confidence/P(fraud),
    Kulczynski=(confidence + fraud recall)/2. Covered IDs include both labels
    and retain input order; data, binner and rules are not modified.
    """
    encoded = _encode(data, fitted_binner)
    results, seen = [], set()
    for rule in rules:
        if rule["rule_id"] in seen:
            raise ValueError("Rule IDs must be unique.")
        seen.add(rule["rule_id"])
        if rule["consequent"] != {"Class": 1} or not rule["antecedent"]:
            raise ValueError("Rules must have a feature antecedent and Class=1 consequent.")
        mask, used = np.ones(len(data), dtype=bool), set()
        for item in rule["antecedent"]:
            feature, value = item["feature"], item["bin"]
            if feature not in encoded or feature in used:
                raise ValueError("Rule features must be distinct fitted features.")
            if (isinstance(value, bool) or not isinstance(value, Integral)
                    or not 0 <= value < fitted_binner["features"][feature]["effective_bins"]):
                raise ValueError("Rule refers to an invalid fitted bin.")
            used.add(feature)
            mask &= encoded[feature] == value
        results.append({"rule_id": rule["rule_id"], "antecedent": rule["antecedent"],
                        "consequent": {"Class": 1}, **_metrics(data, mask)})
    return results

