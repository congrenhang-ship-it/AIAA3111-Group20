"""负责人 B：监督分类与类别不平衡实验，建议投入 25%。

任务：四种基础模型、有限调参、代表模型的类别权重/训练集重采样。
输入：A 提供的训练特征和标签；输出：拟合流程及对原始行序的风险分数。
公共评价交给 C。不得以测试表现调参；GaussianNB 不支持 class_weight。
状态：训练、打分、类别权重与训练集随机欠采样可用；实验见 run_b_validation.py。
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_consistent_length


MODEL_NAMES = ("logistic_regression", "decision_tree", "gaussian_nb", "random_forest")
STRATEGIES = ("original", "class_weight", "random_undersample")


def _feature_columns(X):
    """Select named features at fit time, excluding label and row metadata.

    For a numeric matrix, every column is a feature: callers must remove
    metadata before converting a DataFrame to an array.
    """
    if hasattr(X, "columns"):
        if not X.columns.is_unique:
            raise ValueError("Feature column names must be unique.")
        columns = [column for column in X.columns if column not in ("Class", "row_id")]
        if not columns:
            raise ValueError("No feature columns remain after excluding Class and row_id.")
        return columns
    array = np.asarray(X)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("X must be a 2-D matrix with at least one feature column.")
    return list(range(array.shape[1]))


def train_model(X_train, y_train, model_name, strategy="original", seed=42):
    """Fit a baseline pipeline using only the supplied training rows.

    X_train is a numeric DataFrame or a 2-D numeric matrix. Named Class and
    row_id columns are excluded inside the pipeline. y_train must be aligned
    by position and contain both numeric labels 0 and 1.

    Logistic regression and GaussianNB use training-fitted StandardScaler;
    trees use the selected features without scaling. Random undersampling
    keeps every minority row and draws an equal number of majority rows
    without replacement, using seed; it never touches validation/test rows.
    Defaults are starting settings, not parameters selected on real data.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name {model_name!r}; choose from {MODEL_NAMES}.")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}.")
    if strategy == "class_weight" and model_name == "gaussian_nb":
        raise ValueError("GaussianNB does not support class_weight.")
    labels = np.asarray(y_train)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("y_train must be a 1-D array containing only numeric labels 0 and 1.")
    if np.unique(labels).size != 2:
        raise ValueError("Training data must contain both Class=0 and Class=1.")
    check_consistent_length(X_train, labels)

    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=seed)
    elif model_name == "gaussian_nb":
        classifier = GaussianNB()
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)

    if strategy == "class_weight":
        classifier.set_params(class_weight="balanced")

    fit_X, fit_y = X_train, labels
    positions = np.arange(len(labels))
    if strategy == "random_undersample":
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(labels == label) for label in (0, 1)]
        count = min(map(len, groups))
        positions = np.sort(np.concatenate([
            rng.choice(group, count, replace=False) if len(group) > count else group
            for group in groups
        ]))
        fit_X = X_train.iloc[positions] if hasattr(X_train, "iloc") else np.asarray(X_train)[positions]
        fit_y = labels[positions]

    transformer = StandardScaler() if model_name in ("logistic_regression", "gaussian_nb") else "passthrough"
    preprocessing = ColumnTransformer(
        [("features", transformer, _feature_columns)], remainder="drop"
    )
    model = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    model.fit(fit_X, fit_y)
    # Positional indices refer only to the supplied training partition.
    model.training_positions_ = positions
    model.training_class_counts_ = {int(label): int((fit_y == label).sum()) for label in (0, 1)}
    return model


def score_model(model, X):
    """Return a finite 1-D Class=1 probability array in X's original row order.

    For predict_proba, locate Class=1 using classes_, not an assumed column.
    Higher means more suspicious. Do not return hard labels or reorder rows.
    Combine with metadata through C's make_prediction_frame.
    """
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if classes.ndim != 1 or positive_columns.size != 1:
        raise ValueError("Model classes_ must contain exactly one Class=1 entry.")
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.shape != (len(X), len(classes)):
        raise ValueError("predict_proba returned a shape inconsistent with X and classes_.")
    if not np.isfinite(probabilities).all():
        raise ValueError("predict_proba returned non-finite probabilities.")
    scores = probabilities[:, positive_columns[0]].copy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Fraud probabilities must be finite and between 0 and 1.")
    return scores


LABEL_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
LABEL_SEEDS = (42, 43, 44)


def nested_label_masks(labels, seed):
    """E03 controlled, label-informed simulation, not real annotation acquisition.

    One random permutation per class; ceil class_count*fraction, minimum one.
    Returned positions are in original training order, never validation rows.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Label simulation requires a binary training vector")
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(np.flatnonzero(y == label)) for label in (0, 1)]
    return {fraction: np.sort(np.concatenate([
        group[:int(np.ceil(len(group) * fraction))] for group in permutations
    ])) for fraction in LABEL_FRACTIONS}


def fixed_error_cases(predictions, k, per_category=2):
    """E05: near-boundary FP/FN and highest-ranked FP, deterministically selected."""
    frame = predictions.sort_values(["score", "row_id"], ascending=[False, True]).copy()
    frame["rank"] = np.arange(1, len(frame) + 1)
    reviewed = frame.iloc[:k]
    selections = {
        "near_boundary_FP": reviewed.loc[reviewed.Class.eq(0)].tail(per_category).iloc[::-1],
        "near_boundary_FN": frame.iloc[k:].loc[lambda x: x.Class.eq(1)].head(per_category),
        "highest_ranked_FP": reviewed.loc[reviewed.Class.eq(0)].head(per_category),
    }
    import pandas as pd
    return pd.concat([part.assign(category=name) for name, part in selections.items()], ignore_index=True)


def cold_start_groups(train, validation):
    """E13 mutually exclusive groups anchored to initial training entities."""
    new_customer = ~validation.CUSTOMER_ID.isin(train.CUSTOMER_ID)
    new_terminal = ~validation.TERMINAL_ID.isin(train.TERMINAL_ID)
    names = np.array(["both_seen", "new_customer_only", "new_terminal_only", "both_new"])
    return names[new_customer.to_numpy(int) + 2 * new_terminal.to_numpy(int)]

