"""负责人 B：监督分类与类别不平衡实验，建议投入 25%。

任务：四种基础模型、有限调参、代表模型的类别权重/训练集重采样。
输入：A 提供的训练特征和标签；输出：拟合流程及对原始行序的风险分数。
公共评价交给 C。不得以测试表现调参；GaussianNB 不支持 class_weight。
状态：B1 基础训练与打分接口已实现；真实数据基线和不平衡实验尚未完成。
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
    trees use the selected features without scaling. Only the original
    strategy is implemented for B1; weighting/resampling belong to B4.
    Defaults are starting settings, not parameters selected on real data.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name {model_name!r}; choose from {MODEL_NAMES}.")
    if strategy != "original":
        raise ValueError("B1 supports strategy='original' only; imbalance strategies are pending B4.")
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

    transformer = StandardScaler() if model_name in ("logistic_regression", "gaussian_nb") else "passthrough"
    preprocessing = ColumnTransformer(
        [("features", transformer, _feature_columns)], remainder="drop"
    )
    model = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    return model.fit(X_train, labels)


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

