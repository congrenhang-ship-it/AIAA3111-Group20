"""负责人 C：聚类（15%）与异常检测（25%），比例为建议工作投入。

任务：K-means 群体探索、最近簇中心距离、Isolation Forest 风险排序。
输入：A 的训练特征，不含 Class/row_id；输出：拟合流程、簇编号和风险分数。
训练只使用混合训练样本的特征，不按标签筛出正常样本。若使用验证标签
选择异常参数或组合权重，需要在报告明确这一点，不宣称全流程无标签。
仅训练阶段拟合缩放器；与 B 使用相同划分，由 evaluation.py 统一评价。
状态：以下接口均未实现；不要将簇编号当作欺诈预测。
"""


def fit_kmeans(X_train, n_clusters=5, seed=42):
    """TODO C: return a fitted scaling + KMeans pipeline.

    Compare only a few k values; select using training structural metrics and
    stability. Compute silhouette on a fixed subset to control resource use.
    Never force k=2 to represent the two target labels.
    """
    raise NotImplementedError("K-means training is not implemented yet.")


def score_kmeans(model, X):
    """TODO C: return (cluster_ids, nearest_center_distances) aligned with X.

    Apply the training scaler and centers without refitting. Higher distance
    means more unusual; cluster labels are separate descriptive metadata.
    """
    raise NotImplementedError("Cluster assignment and distance scoring are not implemented yet.")


def fit_isolation_forest(X_train, seed=42, **params):
    """TODO C: return a fitted Isolation Forest pipeline on mixed training data."""
    raise NotImplementedError("Isolation Forest training is not implemented yet.")


def score_isolation_forest(model, X):
    """TODO C: return finite 1-D scores, higher = more suspicious.

    Invert score_samples (or decision_function) consistently; keep the choice
    documented. Preserve row order and do not return only {-1, 1} labels.
    """
    raise NotImplementedError("Isolation Forest scoring is not implemented yet.")
