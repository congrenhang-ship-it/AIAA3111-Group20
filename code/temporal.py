"""负责人 A：相对时间窗口评价，建议投入 15%。

任务：按测试前确定的窗口分析固定模型，复用 C 的评价函数。
输入：evaluation.py 约定的预测表；输出：各窗口样本/欺诈数及审核指标。
窗口以原始 Time 的共同相对起点划分，所有方法使用同样窗口和预算。
这不是重新训练、调参或构造个人历史；两天数据不支持长期季节性结论。
状态：接口占位，尚未实现。
"""


def evaluate_time_windows(predictions, window_seconds=3600, origin=0, review_fractions=(0.001, 0.005, 0.01, 0.02), threshold=None):
    """TODO A: group by method and floor((Time-origin)/window_seconds).

    Require positive window_seconds and a shared predeclared origin. Pass each
    nonempty window's Class, score and row_id to evaluation.evaluate_scores.
    Keep the same window IDs for all methods, flag empty windows, and report
    positive counts. No-positive windows have undefined AP/Recall, not zero.
    Distinguish per-window budgets from a single global top-K selection.
    """
    raise NotImplementedError("Time-window evaluation is not implemented yet.")
