# B1 分类接口检阅说明

负责人：B。当前为本地待检阅实现，尚未提交 PR；TASKS.md 的 B1 勾选待完成提交与验收后更新。

## 已实现

- `code/classification.py` 保留现有 `train_model`、`score_model` 签名。
- 支持逻辑回归、决策树、Gaussian NB、随机森林四种基础分类器。
- 返回包含特征选择、预处理和分类器的已拟合 Pipeline。
- DataFrame 中的 `Class`、`row_id` 在 Pipeline 内排除；特征列在训练时确定，验证时按列名对齐。
- 支持纯数值二维数组；数组没有列名，调用方必须提前移除标签和编号，并保持特征列顺序及输入格式一致。
- `score_model` 根据 `classes_` 定位标签 1 的概率列，返回与输入行顺序一致的一维数组，并检查长度、有限值和概率范围。
- 标签必须是同时含有 0、1 的一维数值序列，且与训练特征按位置对齐；不按 pandas 索引自动重排。

## 当前参数，供检阅调整

| 模型 | 预处理 | 显式参数 |
|---|---|---|
| logistic_regression | StandardScaler | max_iter=1000，random_state=seed |
| decision_tree | 原始特征 | random_state=seed |
| gaussian_nb | StandardScaler | 使用默认模型参数 |
| random_forest | 原始特征 | n_estimators=100，random_state=seed，n_jobs=1 |

其余参数使用当前 scikit-learn 默认值。上述设置用于起步，尚未经过真实数据验证或调参。缩放器只在训练时拟合，打分时只 transform。

当前 `strategy` 仅接受 `original`。类别权重与重采样属于 B4，其他值会明确报错。

## 使用示例

以下代码供 A 在已有模块导入方式下接入；train/validation 必须来自 A 的唯一划分：

```python
from classification import train_model, score_model
from preprocessing import get_features

model = train_model(
    get_features(train), train["Class"],
    model_name="logistic_regression", seed=42,
)
validation_scores = score_model(model, get_features(validation))
# C 的 make_prediction_frame 就绪后：
# predictions = make_prediction_frame(validation, validation_scores, "lr_original")
```

没有修改公共评价或主入口；它们保持各自负责人的接口。

## 已执行检查

在仓库根目录运行：

```powershell
py -3.12 -m unittest discover -s code -p test_classification.py -v
```

本地结果：6 项测试全部通过。检查使用固定种子生成的 160 行合成数据，其中前 120 行用于训练、后 40 行用于打分。这一划分只检查程序，正式实验必须使用 A 的时间划分。

检查覆盖：四种分类器、行序与列序变化、训练及打分阶段的标签/编号排除、打分时不重新拟合缩放器、数值数组输入、反转的 classes_ 顺序，以及非法标签和输出。

本地检查环境：Python 3.12，NumPy 2.4.4，pandas 3.0.2，scikit-learn 1.8.0。这里只记录实际检查环境，未修改全组依赖要求。

## 后续 B2

当前仓库没有 `data/creditcard.csv`，`split_data` 仍为占位函数。A3 完成后，在同一时间划分上运行逻辑回归，记录命令、参数、训练耗时、数据版本与代码 commit；对验证集输出风险分数，交给 C 的公共评价。以上合成检查不构成 B2/B3 真实实验结果。
