"""Priority 1: common metrics, review-budget evaluation and plotting interfaces."""


def evaluate_scores(y_true, scores, review_fractions=(0.001, 0.005, 0.01, 0.02)):
    """TODO: calculate AP and top-K review counts, Precision@K and Recall@K.

    Higher scores must mean higher fraud risk. Define K rounding and tie
    handling consistently. Preserve the evaluation set's natural class ratio.
    Select fixed prediction thresholds on validation data, not test data.
    """
    raise NotImplementedError("Shared evaluation is not implemented yet.")


def save_figures(results, output_dir):
    """TODO: save PR curves, review-budget curves and error-analysis figures."""
    raise NotImplementedError("Experiment plotting is not implemented yet.")

