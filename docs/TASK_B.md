# B 分类实验协议与交接

本协议在首次真实验证比较前写入。B 使用本地 `b/classification-interface` 分支，已同步 A 的 `origin/Task_A`（26c1c32）和 C 的 `origin/codex/c-independent-modules`（24c9212）。远程 main 当前尚不含这些实现。

## 预先限定的实验

1. 四种原始基线：逻辑回归、决策树、Gaussian NB、随机森林。
2. 只对逻辑回归比较 `class_weight='balanced'` 和一次训练集随机欠采样。
3. 欠采样保留全部少数类，从多数类无放回抽取同样数量，seed=42；保持抽取后的训练行原顺序。缩放器在该抽样训练部分拟合，验证集保持自然比例。此处比较的是完整训练策略。
4. 不额外扩大参数搜索。逻辑回归 max_iter=1000，随机森林 100 棵树；其余参数完整记录在结果 JSON 中。
5. 只使用 A 的原始数据校验、去重和时间划分。标签及 row_id 不进入特征。测试部分只核对分段身份，不打分。

逻辑回归和 Gaussian NB 用 StandardScaler，树模型不缩放。所有方法输出按验证行序排列的 Class=1 概率。

## 方案选择

沿用 C 的主要审核预算 1%，并报告 0.1%、0.5%、1%、2%。复用 `evaluate_scores`，K=ceil(N×q)，按 score 降序、row_id 升序稳定排名。选择验证集 1% 预算下 TP 最大的配置；并列时 AP 较高者优先，仍并列按方法名称升序。

操作决策直接使用 top-K，固定概率阈值记为 null。只用概率阈值无法保证在分数并列时恰好审核 K 笔，因此不另选阈值来替代预算排名。模型仅在训练段拟合，冻结后不合并训练和验证段重训。

## 命令与输出

在仓库根目录执行：

```powershell
py -3.12 code/main.py --prepare-data
py -3.12 code/main.py --run-validation
py -3.12 code/run_b_validation.py
py -3.12 -m unittest discover -s code -p test_classification.py -v
py -3.12 -m unittest discover -s tests -v
```

- `results/tables/b_validation.json`、`b_validation.csv`：全部验证实验的参数、耗时、公共指标和源代码指纹。
- `results/tables/b_freeze.json`：B 的选择配置，待全组审核后纳入 G2；不代表全组 G2 已通过。
- `data/b_validation/`：本地完整验证预测、公共评价详情、选中模型和选中验证预测。不提交到 Git。
- `data/b_validation/selected_validation_predictions.csv`：交给 C 的 `--classification-predictions`，列名为 row_id/Class/Time/method/score。
- `data/b_validation/selected_model.joblib`：本地保留训练好的流程供 A 在 G2 后评分。脚本检查保存后重新加载的分数在数值容差内一致，且全部审核排序完全一致。

A 可按冻结 JSON 中的 model_name/strategy/seed，调用原来的 train_model 接口从同一训练段重建模型。C 的验证互补分析可用 `py -3.12 code/main.py --run-c --classification-predictions data/b_validation/selected_validation_predictions.csv`，须遵守 C 的已有验证流程。

## 解释与后续验收

AP、Precision@K 和 Recall@K 为主要证据。严重不平衡时，单看准确率不足以说明审核价值。类别权重改变错误代价；欠采样减少多数类训练信息，可能损害排序。未校准的概率不能直接解释为可靠的真实欺诈概率。

B2-B5 的本机运行不等同于另一台电脑上的 G1 复现；需由组员核对公共输出。B6 的正式测试案例和 B7 的最终报告数字依赖 A 的全组冻结测试。验证结果、候选选择与正式测试结论必须分开记录。

本机现有 Python 包版本与 A 固定的 requirements.txt 存在差异。脚本记录实际 Python/NumPy/pandas/scikit-learn/SciPy 版本；候选方案需在共同版本环境复核后用于全组最终测试。未更改用户现有运行环境。
