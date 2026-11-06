"""Priority 1: shared classification training interface, not yet implemented."""


MODEL_NAMES = ("logistic_regression", "decision_tree", "gaussian_nb", "random_forest")


def train_model(X_train, y_train, model_name, strategy="original", seed=42):
    """TODO: fit preprocessing and a classifier using training data only.

    Compare the four baseline models before adding class weighting and one
    resampling method to representative models. GaussianNB does not accept
    the same class_weight option as logistic regression and tree models.
    Return the fitted pipeline; it must score unseen validation/test rows.
    """
    raise NotImplementedError("Classification experiments are not implemented yet.")

