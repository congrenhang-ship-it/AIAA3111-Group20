# C 模块：实现、运行和报告素材

负责人：C。本文记录可以独立完成的工作，并供 A 接入、B 使用评价、D 整理报告。合成检查不是课程实验结果。

## 1. 完成情况

| 任务 | 当前交付 | 正式完成还需要什么 |
|---|---|---|
| C1 | 统一预测表、输入检查、确定性排名规则已完成 | 无 |
| C2 | AP、PR、预算指标、阈值计数、随机期望基线已完成，并通过人工例子检查 | 无；G1 仍需要 A/B 连通和第二台电脑复现 |
| C3 | K-means、训练结构比较、固定子样本轮廓系数、种子稳定性、验证簇分配和距离已实现 | A3 的正式划分，真实数据运行与分析 |
| C4 | Isolation Forest 管线、统一分数方向、同预算比较入口已实现 | A3 的正式划分，真实异常排序实验 |
| C5 | 所有方法的分群评价、名单交集、固定容量组合与补齐已实现 | A3 和 B5 的分类预测；正式互补性分析 |
| C6 | 参数与规则候选、运行溯源记录已准备 | 验证结果、共同主预算和 B5；不能提前宣称冻结 |
| C7 | PR、预算 Precision/Recall、群体比例与检出表现、等预算检出量五类图已实现 | A8 的最终测试预测和方案冻结记录 |
| C8 | 下方英文方法稿、结果写作提纲、问答准备已完成 | 真实结果及其解释、D 的报告整合 |

当前没有正式数据划分或分类结果，本次没有替 A/B 完成相应任务，也没有读取或评价最终测试集。

## 2. 可以直接运行

在仓库根目录激活环境。C 模块需要 numpy、pandas、scikit-learn、matplotlib；全组依赖仍以根目录 requirements.txt 为准。

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python code/run_c_validation.py --demo
```

合成演示默认保存在 `data/c_demo/`，标题明确标出 SYNTHETIC DEMO，不上传 GitHub，不放进正式结果。重复运行时使用新的输出目录，以保留旧记录：

```bash
python code/run_c_validation.py --demo --output data/c_demo_run2
```

本机已检查 Python 3.12.14 环境，核心包为 numpy 2.5.4、pandas 3.1.1、scikit-learn 1.9.1、matplotlib 3.11.2。各次运行的实际版本、代码 SHA-256、commit 和工作区是否有未提交修改，均写入 JSON；这些版本是本机检查记录，尚未代替 A1 的共同环境冻结。

### 收到 A3 后

A 从唯一划分导出 `data/train.csv`、`data/validation.csv`。每份保留 `row_id, Class, Time, V1,...,V28, Amount`，原始数值不提前全局缩放；row_id 为原始 CSV 行编号且唯一。两部分 ID 无交集，最大训练 Time 严格小于最小验证 Time，同一时间不跨边界。CSV 文件名仅为接口建议，A 可以指定其他路径。

```bash
python code/run_c_validation.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --output data/c_validation_run1
```

无需等 B 完成，先运行异常与聚类验证。收到 B5 后，用同一验证集的标准预测表运行：

```bash
python code/run_c_validation.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --classification-predictions data/classifier_validation.csv \
  --output data/c_validation_with_classifier
```

分类 CSV 必须只有一种 method；名称使用英文字母、数字、下划线或连字符。预测行可以乱序，程序按 row_id 对齐并核查标签和 Time。输入的划分身份由 A 保证：程序能够检查时间和 ID，但不能仅凭文件内容判断某个文件是否实际上是测试集。

每次运行产出：完整预测 CSV、训练结构比较 CSV、验证预算表、群体统计表、各簇特征均值表、五类图（无分类输入时四类），以及含配置、版本、输入文件 SHA-256 和指标的 `validation_results.json`。结果写在 Git 忽略的 data/ 内；正式验证后选择小型汇总放入 results/，完整预测保留本地。

### 给 A/B 的最小调用方式

```python
from preprocessing import get_features
from unsupervised import fit_kmeans, score_kmeans, fit_isolation_forest, score_isolation_forest
from evaluation import make_prediction_frame, evaluate_scores, evaluate_groups, compare_review_lists

X_train, X_val = get_features(train), get_features(validation)
km = fit_kmeans(X_train, n_clusters=5, seed=42)  # 5 只是调用示例，不是冻结结论
cluster_ids, distances = score_kmeans(km, X_val)
forest = fit_isolation_forest(X_train, seed=42)
scores = score_isolation_forest(forest, X_val)
pred = make_prediction_frame(validation, scores, 'isolation_forest', cluster_ids)
metrics = evaluate_scores(pred.Class, pred.score, row_ids=pred.row_id)
groups = evaluate_groups(pred, cluster_ids, [0.001, 0.005, 0.01, 0.02])
# classifier_pred 是 B 对同一 validation 的标准预测表；k 是共同审核容量
# comparison = compare_review_lists(classifier_pred, pred, k, classifier_share=0.5)
```

B 输出连续分数后直接调用 make_prediction_frame 和 evaluate_scores，不需要另写评价。A 的时间窗口调用同一个 evaluate_scores；无正类/空窗口的 None 在报告显示为 N/A，不改成 0。

## 3. 已固定的接口与候选设置

- **预测表**：row_id、Class、Time、method、score，可额外保存 cluster_id。分数数组按位置对应输入，避免 pandas 索引重排。非法标签、重复/非整数 ID、非有限分数和长度不符直接报错。
- **排序与预算**：score 降序、row_id 升序；`K = ceil(N × q)`，q 默认 0.1%、0.5%、1%、2%。图按实际 K/N 展示，尤其小组取整后不应继续声称容量恰为 q%。主预算和选优规则仍须全组在验证阶段确定。
- **AP**：使用 sklearn 的 average_precision_score，不是梯形积分 PR-AUC。AP 对同分按同一阈值组处理；top-K 在容量边界用 row_id 打破并列，两者不应混淆。无正类时 AP/Recall 为 None，空集 Precision 也为 None。
- **随机基线**：公共表给出 E[TP]=K×欺诈比例、E[Precision]=欺诈比例、E[Recall]=K/N；它们是期望，不取整为实测次数。另保存 seed=42 的一次随机排序作辅助展示，不把单次随机结果当理论期望。
- **缩放与聚类**：每个管线只在混合训练特征上拟合 StandardScaler，排除 Class/row_id 等元数据；目前使用全部既定 30 个特征，包括相对 Time 和 Amount。Time 可能影响簇结构，需在真实结果中讨论，不能给匿名 V 特征虚构业务含义。验证/测试只 transform/predict。
- **k 候选**：3、5、8，种子 42/43，n_init=10；所有候选用同一训练子样本（最多 1,000 笔）计算轮廓系数，用 ARI 比较两种种子的分群一致性。默认选择平均有效训练轮廓系数最高的 k，并列选小 k；最终拟合种子 42。ARI 是结构稳定性诊断，不是欺诈准确率。退化结构不能计算轮廓系数时记空；全部无效则中止调查数据。
- **距离异常分数**：标准化空间中到最近已训练簇中心的欧氏距离，越大越可疑；不把簇编号当风险大小。
- **Isolation Forest 候选**：100 棵树，max_samples=256，contamination=auto，seed=42，n_jobs=1。使用 `-score_samples`，不以 predict 的硬标签替代排序，不把分数解释成欺诈概率。训练不使用标签，也不先筛出正常样本。上述是可复现的起始配置，并非已用真实验证证实最优。
- **互补性**：两张纯方法 top-K 名单报告重合/各自独有的欺诈。组合先取 `floor(K×分类配额)` 笔分类排序，再沿异常排序去重补齐至 K；候选分类配额 0、25%、50%、75%、100%。取整可能让小 K 的配额重复，应按实际名单解释。组合选择不用标签；配额的验证选优会用标签，报告须披露。候选遍历不是冻结；50/50 图仅为展示，不自动认定最佳。
- **群体分析**：相同 K-means 簇用于所有方法，分别保存组内预算和全局 top-K 在各组的分布。组内 ceil 配额相加可能超过一个全局 K，二者不可直接当同预算比较。
- **冻结后测试**：A 使用冻结的训练管线对测试特征打分，复用评价与绘图；当前独立入口只处理训练/验证，避免替 A 开始最终测试。若决定 train+validation 重拟合，必须事先统一协议，不能静默改变中心或缩放器。

## 4. 可供 D 整合的英文方法稿

以下只写已实现的方法。正式报告应在 G2 后核对实际采用的配置，并引用对应软件/算法来源；不将本节当作已经完成真实实验的证明。

### Clustering and unsupervised anomaly scores

We implement K-means to describe transaction groups without using fraud labels during fitting. StandardScaler is fitted on the mixed training partition and reused for subsequent scoring. The candidate cluster counts are 3, 5, and 8, with two random seeds and ten initializations per fit. Structural quality is assessed using silhouette scores on one fixed training subsample of up to 1,000 transactions. Adjusted Rand index is used to examine agreement between seeds. The validation runner selects the cluster count with the highest mean defined training silhouette score, breaking ties in favour of the smaller count. Fraud labels are used only after fitting to describe group sizes, prevalence, coverage, and detection performance. Cluster identifiers are not treated as fraud predictions.

We also implement two continuous anomaly scores: Euclidean distance to the nearest trained cluster centre in standardized feature space, and the negative of Isolation Forest's score_samples output. The initial Isolation Forest configuration uses 100 trees, max_samples of 256, and random seed 42. Both models are trained on the mixed training features without label-based filtering. Larger scores indicate greater anomaly, rather than calibrated fraud probabilities.

### Evaluation under a review budget

All methods share transaction identifiers and one evaluation implementation. For a review fraction q and N transactions, the review count is K = ceil(Nq). Transactions are ranked by decreasing risk score, with increasing original row identifier breaking ties. We report average precision, precision–recall curves, detected fraud counts, Precision@K, and Recall@K. The expected performance of uniformly random review is reported separately from a seeded random ranking. AP and Recall are marked as not applicable when a partition contains no fraudulent transactions.

To assess complementarity, we compare the fraud cases shared by the two pure top-K lists and those unique to either list. A capacity-preserving combined policy first selects floor(Kα) transactions from the classifier ranking, then scans the anomaly ranking, skipping duplicates until exactly K distinct transactions are selected. Candidate values of α are 0, 0.25, 0.5, 0.75, and 1. Selection of a policy using validation labels must be disclosed. Within-group budgets are reported separately from the distribution of a single global top-K list across groups. Final test evaluation requires all configuration choices to be frozen first.

## 5. 真实结果到位后的写作顺序

1. 列训练/验证样本量、欺诈数、实际 k 和结构稳定性；说明这些群体是否有可描述的差异。
2. 比较同预算下分类、Isolation Forest、簇距离及随机期望，不用总体准确率掩盖稀有正类。
3. 先看纯名单的独有欺诈，再看容量固定组合是否真的多检出；独有样本存在并不保证组合改善。
4. 分组报告样本量/正类数和表现，指出小簇的不稳定性。联系 A 的时间结果，但不自行换时间窗口。
5. 报告未发生的改善、误报代价和匿名特征的解释边界。所有数字引用具体结果文件、划分和 commit。

本次不填写任何真实 AP、检出量、最佳方法、最佳配额或漂移结论。

本机验证记录：21 项 unittest 全部通过；合成训练/验证入口成功生成五类图，并逐张检查标题、坐标与图例。检查覆盖人工 AP/阈值计数、并列排序、零正类/空集、行身份对齐、重叠名单补齐、不依赖标签选名单、训练边界、距离公式、异常方向、结构稳定性和 CLI 输出保护。此记录不代表第二台电脑复现或真实 ULB 实验通过。

## 6. 讲解与答辩准备

- **为什么不是两个簇？** 正常交易也可能形成多个结构，欺诈也不一定集中在一个簇；k 用训练结构比较，不能预设与标签一一对应。
- **为什么异常分不等于欺诈概率？** 它衡量与训练结构的偏离，少见的正常交易也可能很异常；最终必须用留出标签评价。
- **为什么只比较 top-K？** 审核人员容量固定，多查一倍自然可能多找出欺诈，不能证明方法更有效。
- **为什么同时给 AP 和预算指标？** AP描述整体排序；预算指标回答实际只能审核少量交易时能找到多少，两者可能给出不同排序。
- **为何组合不直接求并集？** 两张 top-K 的并集容量可能接近 2K；固定配额去重补齐才能在相同容量下比较。
- **为什么不保证组合更好？** 异常独有欺诈可能伴随更多误报，还会占用分类的容量；必须看同 K 的真实结果。
- **是否完全没有用标签？** 无监督模型拟合和 k 的结构比较不使用标签；评价、画像以及若采用验证选配额则使用留出标签，应明确区分。
- **程序检查能证明实验效果吗？** 不能。人工例子/合成数据检查计算和管线，真实有效性来自 A 的固定划分及最终留出评价。

## 7. 参考实现依据

- [sklearn: average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [sklearn: IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [sklearn: silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [sklearn: adjusted_rand_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

代码与方法稿由 C 使用 AIGC 辅助编写；本次检查在单台电脑上完成。最终 Disclosures 由全组据实际使用情况核对，第二台电脑复现尚未完成。
