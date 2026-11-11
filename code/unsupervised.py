"""负责人 C：训练集 K-means、簇中心距离、Isolation Forest。

已实现，不使用训练标签。所有缩放仅在训练时拟合；打分不重排样本。
聚类结构选择与稳定性仅依赖训练特征；验证标签调参须在报告披露。
"""

import itertools
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _features(X, allow_empty=False):
    forbidden = {"Class", "row_id", "cluster_id", "method", "score"}
    if hasattr(X, "columns"):
        if not X.columns.is_unique or forbidden & set(X.columns):
            raise ValueError(
                "Inputs must have unique feature columns, excluding target/metadata"
            )
    try:
        a = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric") from exc
    if (
        a.ndim != 2
        or not a.shape[1]
        or (not allow_empty and not len(a))
        or not np.isfinite(a).all()
    ):
        raise ValueError(
            "Features must be a finite nonempty 2-D matrix; resolve missing values with A"
        )
    return X


def fit_kmeans(X_train, n_clusters=5, seed=42):
    """Fit StandardScaler + KMeans with explicit deterministic initialization."""
    _features(X_train)
    if (
        isinstance(n_clusters, bool)
        or int(n_clusters) != n_clusters
        or not 2 <= n_clusters <= len(X_train)
    ):
        raise ValueError("n_clusters must be an integer between 2 and training size")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=int(n_clusters), n_init=10, random_state=seed)),
        ]
    ).fit(X_train)


def score_kmeans(model, X):
    """Return cluster IDs and distances in the fitted standardized feature space."""
    _features(X, allow_empty=True)
    if not len(X):
        return np.array([], dtype=int), np.array([], dtype=float)
    transformed = model.named_steps["scaler"].transform(X)
    estimator = model.named_steps["model"]
    clusters = estimator.predict(transformed)
    distances = np.linalg.norm(
        transformed - estimator.cluster_centers_[clusters], axis=1
    )
    return clusters, distances


def fit_isolation_forest(X_train, seed=42, **params):
    """Fit on mixed training features only, without Class or label-based filtering."""
    _features(X_train)
    if "random_state" in params:
        raise ValueError("Use seed rather than passing random_state twice")
    settings = {
        "n_estimators": 100,
        "max_samples": 256,
        "contamination": "auto",
        "n_jobs": 1,
    }
    settings.update(params)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", IsolationForest(random_state=seed, **settings)),
        ]
    ).fit(X_train)


def score_isolation_forest(model, X):
    """Negate score_samples so higher is more suspicious; do not use hard labels."""
    _features(X, allow_empty=True)
    if not len(X):
        return np.array([], dtype=float)
    return -model.score_samples(X)


def compare_kmeans_candidates(
    X_train, ks=(3, 5, 8), seeds=(42, 43), sample_size=1000, progress=None
):
    """Return structural metrics and seed stability on ONE fixed training subset.

    Candidate selection can use mean defined silhouette with smaller k as tie
    break. Degenerate samples get None, never fabricated silhouette. No labels
    or validation rows enter this routine. Pairwise ARI measures seed stability.
    """
    _features(X_train)
    ks, seeds = tuple(ks), tuple(seeds)
    if (
        not ks
        or len(set(ks)) != len(ks)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Provide distinct ks and at least two distinct seeds")
    if (
        isinstance(sample_size, bool)
        or int(sample_size) != sample_size
        or sample_size < 2
    ):
        raise ValueError("sample_size must be an integer >= 2")
    rng = np.random.default_rng(42)
    indices = np.sort(
        rng.choice(
            len(X_train), size=min(int(sample_size), len(X_train)), replace=False
        )
    )
    sample = (
        X_train.iloc[indices]
        if hasattr(X_train, "iloc")
        else np.asarray(X_train)[indices]
    )
    records = []
    for k in ks:
        assignments = []
        for seed in seeds:
            if progress is not None:
                progress(f"K-means training: k={k}, seed={seed}")
            pipeline = fit_kmeans(X_train, k, seed)
            labels, _ = score_kmeans(pipeline, sample)
            transformed = pipeline.named_steps["scaler"].transform(sample)
            count = len(np.unique(labels))
            silhouette = (
                float(silhouette_score(transformed, labels))
                if 1 < count < len(sample)
                else None
            )
            assignments.append(labels)
            records.append(
                {
                    "k": int(k),
                    "seed": int(seed),
                    "inertia": float(pipeline.named_steps["model"].inertia_),
                    "sample_size": len(sample),
                    "sample_silhouette": silhouette,
                }
            )
        stability = float(
            np.mean(
                [
                    adjusted_rand_score(x, y)
                    for x, y in itertools.combinations(assignments, 2)
                ]
            )
        )
        for row in records[-len(seeds) :]:
            row["seed_stability_ARI"] = stability
    return records


def score_frozen_unsupervised(bundle, X):
    """A's final scoring hook: reuse persisted training pipelines, never select/refit.

    bundle is the local joblib artifact written by run_c_validation. Load only
    your own trusted artifact. Caller supplies features only; labels are unused.
    Whole-group G2 and permission to evaluate the test partition belong to A.
    """
    _features(X, allow_empty=True)
    config = bundle["config"]
    if hasattr(X, "columns") and list(X.columns) != config["feature_columns"]:
        raise ValueError("Frozen feature order does not match scoring input")
    if bundle["kmeans"].named_steps["model"].n_clusters != config["n_clusters"]:
        raise ValueError("Frozen K-means configuration disagrees with fitted model")
    groups, distance = score_kmeans(bundle["kmeans"], X)
    return {
        "cluster_ids": groups,
        "kmeans_distance": distance,
        "isolation_forest": score_isolation_forest(bundle["isolation_forest"], X),
    }
