"""负责人 B：监督分类与类别不平衡实验，建议投入 25%。

任务：四种基础模型、有限调参、代表模型的类别权重/训练集重采样。
输入：A 提供的训练特征和标签；输出：拟合流程及对原始行序的风险分数。
公共评价交给 C。不得以测试表现调参；GaussianNB 不支持 class_weight。
状态：接口占位，尚未实现训练与打分。
"""


MODEL_NAMES = ("logistic_regression", "decision_tree", "gaussian_nb", "random_forest")


def train_model(X_train, y_train, model_name, strategy="original", seed=42):
    """TODO: fit preprocessing and a classifier using training data only.

    Compare the four baseline models before adding class weighting and one
    resampling method to representative models. GaussianNB does not accept
    the same class_weight option as logistic regression and tree models.
    Return the fitted pipeline; it must score unseen validation/test rows.
    """
    raise NotImplementedError("Classification experiments are not implemented yet.")


def score_model(model, X):
    """TODO B: return a finite 1-D fraud score array aligned with X's row order.

    For predict_proba, locate Class=1 using classes_, not an assumed column.
    Higher means more suspicious. Do not return hard labels or reorder rows.
    Combine with metadata through C's make_prediction_frame.
    """
    raise NotImplementedError("Classification scoring is not implemented yet.")

