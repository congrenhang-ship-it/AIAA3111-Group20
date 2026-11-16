# B 分类方法与验证材料

更新：2026-11-16。本文供 D 整合方法与验证部分；全部为 validation，尚无正式 test 结论。

当前状态：main@2750902 已验收 B 的 E03/E05/E10/E13，主线记录 91 项公共检查和 11 项分类检查通过；历史段落中的依赖待复核事项已解除。下文保留原始运行记录，末尾提供 B6 可排版英文正文和 B7 讲解准备。正式测试仍待 A7/G2、A8。

## 已有 ULB 基线与不平衡对照

四种基础模型和两种不平衡对照见 results/tables/b_validation.json。固定候选为带权逻辑回归，StandardScaler 仅拟合训练数据，class_weight=balanced、max_iter=1000、seed=42。主预算为 1%，score 降序、row_id 升序，K=ceil(Nq)。原验证段 56,745 笔/57 笔欺诈，固定模型 AP=0.772681，检出 49/57，审核 568 笔。原模型和组合不因本轮扩展重新选优。

## E03 标签不足

每种子按真实训练类别分别随机排列，比例取每类 ceil(n×fraction)，构造嵌套子集。该分层过程使用真实标签，是受控模拟，并非现实标注过程。隐藏标签从分类训练中移除，绝不改成正常；缩放和 LR 仅拟合已标注子集。模型种子固定 42，掩码种子为 42/43/44。100% 的三个相同子集复用同一次拟合。

| 标签比例 | 实际样本/欺诈 | AP 均值 [最小,最大] | Recall@1% 均值 [最小,最大] |
|---|---:|---:|---:|
| 5% | 8513/18 | 0.3799 [0.2562,0.6259] | 69.01% [59.65%,78.95%] |
| 10% | 17025/35 | 0.4403 [0.2847,0.6324] | 71.93% [68.42%,77.19%] |
| 25% | 42560/86 | 0.7389 [0.7248,0.7533] | 83.04% [80.70%,84.21%] |
| 50% | 85118/171 | 0.7648 [0.7485,0.7733] | 84.80% [84.21%,85.96%] |
| 100% | 170235/342 | 0.7727 [0.7727,0.7727] | 85.96% [85.96%,85.96%] |

固定 IF256 参考 AP=0.016546。IF 使用全部无标签训练特征、StandardScaler、100 棵、max_samples=256、contamination=auto、seed=42；它与 LR 的可见样本量不同，因此曲线不能归因于算法本身。三种子范围不是置信区间。少标签时结果受具体抽中欺诈样本影响，未预设性能交叉。完整条件、计数、耗时、掩码哈希和配置见 b_extensions.json。

## E05 模型依赖与固定错误案例

对完整验证集的 30 个原始输入逐列置换，AP 评分、seed=42、每列 5 次，无重新拟合或删特征。均值和标准差刻画该固定模型对输入的依赖，不是因果贡献；相关特征会影响置换结果。Amount 重要性最高也不能解释为真实金融业务机制，V 特征仍按匿名名称报告。

| 特征 | 平均 AP 降幅 | 标准差 |
|---|---:|---:|
| Amount | 0.580432 | 0.005234 |
| V2 | 0.167743 | 0.007136 |
| V7 | 0.123094 | 0.003825 |
| V20 | 0.107728 | 0.006890 |
| V14 | 0.105797 | 0.011333 |
| V6 | 0.085556 | 0.006040 |
| V1 | 0.084102 | 0.008369 |
| V5 | 0.081068 | 0.005236 |
| V16 | 0.042168 | 0.004479 |
| V17 | 0.035996 | 0.002088 |

案例规则在运行前写入代码：审核线内最接近边界的两笔误报、线外最接近边界的两笔漏报、最高排名的两笔误报；类别可重叠。逐行身份和原始特征留在本地 CSV，不逐挑“支持故事”的交易。

| 类型 | 全局排名 | 风险分数 |
|---|---|---|
| near_boundary_FP | 568, 567 | 0.804282, 0.804383 |
| near_boundary_FN | 1405, 1740 | 0.594265, 0.536400 |
| highest_ranked_FP | 35, 37 | 1.000000, 0.999999 |

概率没有校准，接近审核线并不等于接近 0.5；排序错误和匿名特征差异只能作诊断，不能建立交易行为的因果解释。全部 30 项数值及五次原值见汇总，前十项见 b_extensions.png。

## Handbook E10 历史消融

按 A 固定版本生成 60 天，30/15/15 天划分；原始数据 SHA-256 以及训练/验证四视图共八个指纹与 A 的交付完全一致。292,139 笔/1,752 笔欺诈训练，145,315 笔/1,291 笔欺诈验证。直接消费 A 的严格过去历史视图，客户/终端编号、标签、时钟原值和生成机制均不入模。四组均重新拟合同一带权 LR，不把 ULB 权重迁移到模拟数据。

| 视图 | AP | TP@1% | Recall@1% | Precision@1% |
|---|---:|---:|---:|---:|
| transaction_only | 0.255353 | 331 | 25.64% | 22.76% |
| customer_history | 0.218989 | 348 | 26.96% | 23.93% |
| terminal_history | 0.235494 | 325 | 25.17% | 22.35% |
| both_history | 0.208312 | 342 | 26.49% | 23.52% |

两类历史 AP 低于单笔视图，但 1% 审核下的检出量略多；客户历史的检出数更高也不触发重新选择后续参考视图，两类历史仍为协议指定参考。模拟器的金额/实体机制以及线性模型限制可能影响结果，不能把历史特征预设成提升，也不证明现实泛化。金额覆盖及全部预算计数保存在 handbook_b.json，不跨数据集平均 AP。

## E13 冷启动

以初始训练实体集合固定四个互斥组，历史可用率按客户/终端过去 7 天是否存在记录定义。复用 C 的同一个全局 top-K 分群分配，未把每组独立配额相加；各组审核量与检出量已核对可加总。

| 群体 | 交易数 | 欺诈数 |
|---|---:|---:|
| both_seen | 145296 | 1291 |
| new_customer_only | 19 | 0 |
| new_terminal_only | 0 | 0 |
| both_new | 0 | 0 |

新客户组只有 19 笔且无欺诈，新终端相关两组为空；AP/Recall 如实记为 N/A。该样本无法支持冷启动检出能力比较，也不是人口公平性结论，没有重新生成数据来凑群体。四个静态模型的组内 AP、历史可用率和全局名单分配保存在同一汇总。

## 运行、来源与剩余验收

使用现有脚本：

```powershell
python -X utf8 code/run_b_validation.py --extensions --output data/b_extensions_20261115_r2
python -X utf8 code/run_b_validation.py --handbook --output data/handbook_b_20261115
python -X utf8 -m unittest discover -s code -p test_classification.py -v
```

B 的 11 项检查通过。全仓 85 项中 84 项通过，test_B_reproduction_checks_identity_parameters_and_persistence 在共同依赖版本门槛处拒绝运行；本机仍为 NumPy 2.4.4、pandas 3.0.2、scikit-learn 1.8.0、SciPy 1.17.1，与 A 锁定版本不同。没有绕过该门槛，也没有把全仓检查写成通过。模型环境、代码指纹、模型/掩码哈希和运行命令均保留，最终冻结前需共同环境复核。

本轮修复：Handbook 首次生成遇到 Windows 相对目标重命名失败，使用绝对输出路径后成功，未改 A 的算法；ULB 首次导出遇到相对路径与绝对根目录比较错误，规范输出路径后完整重跑相同条件。结果没有用于调整模型、参数或生成配置。

B-E1/E2/N1/N2 已实现并完成本机正式数据验证，供 A7/D 接入；正式验收还包括共同依赖复核。B6 现有方法和验证稿已可整合，正式 test 数字及 B7 最终核对等待 A7/G2 和 A8。未评分任何测试段。Codex 辅助实现、检查和文字整理，应按实际情况纳入 Disclosures。

## 2026-11-16 指纹修复与 Handbook 共同环境复核

上面的运行环境限制是 11 月 15 日的原始交付记录。此次在共同锁定环境重新运行 Handbook 四个 LR 及冷启动评价，四组指标、金额统计和冷启动汇总与原结果完全一致；新 handbook_b.json 已记录实际环境。B/C 使用同一 `handbook-view-v2-time12` 指纹函数，八个视图指纹完全匹配。该函数仅在计算指纹时规范两列时间编码，不改变模型输入。

88 项公共检查及 11 项 B 检查均通过。Handbook 的 B-N1/N2 共同环境复核已完成；ULB 的 E03/E05 本次未重跑，仍保留其原依赖复核事项。最小修复和跨架构排查证据见 C_IMPROVEMENTS.md 及 handbook_c_readiness.json；无新增模型搜索或正式测试。


## 2026-11-16 ULB 统一入口接入与完整验收

E03/E05 已接入 python code/main.py --run-team-validation --output data/team_validation。原独立命令继续复用同一函数；统一入口直接传入 A 的训练/验证划分，不重新加载或划分测试数据。所有模型、15 份掩码及逐行案例保存在运行目录的 b_extensions/；聚合表和组合图位于 aggregate/ 原布局。

在锁定环境完整运行后，15 个条件、掩码哈希及 IF 指标与原交付精确一致；E05 全部 150 个置换数值、案例规则和计数一致。错误案例排名仍为 568/567、1405/1740、35/37。13 个独立 LR 模型及 IF 保存后复评分一致，LR 缩放均值核对为各自的已标注训练子集。91 项公共检查＋11 项 B 检查通过。此前 ULB 待共同环境复核的事项现已完成，A6 和 B-E1/E2 不再列为剩余任务。

统一摘要和 B 汇总已写回原 results/tables/a_team_validation.json、b_extensions.json，pending_extensions=[]；没有增加模型搜索或评分正式测试。B6 正式结果补充、B7 及全组 A7/A8/G3 继续保留。Codex 协助接口接入、验证和记录整理，组员按课程要求核对并披露实际协助。

## B6 可排版英文正文：方法与验证结果

本节可由 D 分配至报告的 Methods、Validation Results 和 Limitations，而非独立按成员分章。预算表使用本稿已有的六模型对照来源和四视图表；详细条件保留补充材料。以下内容只报告验证结论。

### Classification and supervision protocol

We compare logistic regression (LR), a decision tree, Gaussian Naive Bayes and a random forest on the shared chronological ULB split. Each pipeline excludes the target and source-row identifier. StandardScaler is fitted within the training pipeline for LR and Gaussian Naive Bayes; the tree models use unscaled features. We additionally compare balanced class weights and seeded 1:1 random undersampling for LR. All minority training rows are retained during undersampling, with an equal-sized majority subset sampled without replacement. Validation retains its natural class distribution. The classifier is selected by validation true positives at the primary 1% review budget, followed by AP and a deterministic method-name tie-break. The capacity is K=ceil(Nq), with descending scores and ascending source-row identifiers resolving ties.

For the label-availability experiment, class-specific random permutations define nested 5%, 10%, 25%, 50% and 100% labeled subsets, using mask seeds 42/43/44 and a fixed model seed of 42. Each class count is rounded upward. The weighted LR pipeline is fitted only on the labeled subset. Stratification uses the true training labels and is therefore a controlled simulation, not an annotation-acquisition strategy. The fixed IF256 reference sees all unlabeled training features, so the comparison changes available records as well as supervision. Permutation importance uses validation AP, five repeats per original input and seed 42, without model refitting or subsequent feature removal.

### ULB validation findings

Weighted LR achieves AP 0.772681 and detects 49 of 57 frauds from 568 reviews (recall 85.96%, precision 8.63%). Original LR detects 48, and the random forest detects 47 despite its higher AP than original LR. Thus overall ranking quality and performance at a specified capacity can order methods differently. Undersampling also detects 48 frauds but reduces AP to 0.157422; it is not an overall improvement. The one-fraud gain of weighting should be interpreted against the small number of positives.

With 5% labeled training data, mean AP is 0.3799, ranging from 0.2562 to 0.6259 across the three masks; at 25%, mean AP is 0.7389. The full-label AP is 0.7727. These ranges describe sensitivity to the sampled labels and are not confidence intervals. The fixed IF256 reference has AP 0.016546 and recall 17.54% at 1%; it differs from the separately developed IF4096 used in the main anomaly comparison. Amount has the largest mean permutation AP decrease (0.580432), followed by V2 and V7. This describes model reliance, not causal effects or business meanings of anonymous features. Fixed diagnostic cases include high-scoring normal transactions and frauds below the review boundary; uncalibrated scores do not guarantee individual correctness.

### Handbook context and cold-start findings

We fit the same weighted LR separately on four A-provided feature views, using identical training and validation transactions and strictly past-only histories. Source data, partitions and all eight train/validation view fingerprints agree with the shared delivery. Transaction-only, customer-history, terminal-history and both-history AP values are respectively 0.255353, 0.218989, 0.235494 and 0.208312. At the common K=1,454, they detect 331, 348, 325 and 342 of 1,291 validation frauds. Histories therefore do not uniformly improve AP, although customer history increases detection at the declared capacity. Both-history remains the predeclared reference; the ablation does not trigger further view selection.

Cold-start groups are anchored to entities observed in the initial training segment. Of 145,315 validation transactions, 145,296 involve previously seen customers and terminals. Only 19 involve new customers, none are fraudulent, and the two new-terminal groups are empty. AP and recall are undefined for these groups; this dataset provides insufficient evidence about cold-start fraud detection. It cannot support demographic-fairness claims. All findings above concern validation, with formal held-out evaluation pending the group freeze. The simulator's designed mechanisms and the datasets' different prevalences also limit cross-dataset generalization.

### 正文图表与数值来源

| 正文内容 | 可用材料 | 限制 |
|---|---|---|
| ULB 六模型/策略对照 | [b_validation.json](../results/tables/b_validation.json)、[比较表](../results/tables/b_validation.csv) | 原始运行来源保留；统一入口已核对固定带权 LR |
| 标签曲线、置换前十项 | [组合图](../results/figures/b_extensions.png)、[完整汇总](../results/tables/b_extensions.json) | 三种子范围与五次置换标准差是不同统计量 |
| 历史消融、冷启动 | [组合图](../results/figures/handbook_b_ablation.png)、[完整汇总](../results/tables/handbook_b.json) | 四组静态模型，不与 A 的滚动快照混比 |
| 统一接入与共同环境验收 | [主线记录](../results/tables/a_team_validation.json) | 102 项通过为远程主线验收记录，不是本机新运行的检查 |

`Report.md` 仍保留旧的“B 的 E03/E05 await delivery”和“第二数据集属于 future work”段落。D 整合时应按最新主线和本稿替换；旧稿不能作为最终状态依据。B 的正文可先整合，不需要等待 D 的 Handbook 规则结果。

## B7 讲解准备：可核查的回答要点

| 常见问题 | 回答要点 |
|---|---|
| 为什么以 1% 预算选模型，而不直接选最高 AP？ | AP 衡量整体排序；预算限定实际能审核的名单。原始 LR 的 AP 低于 RF，但在 568 个名额中多检出 1 笔，必须遵守预定选择规则。 |
| 类别权重和欠采样有什么区别？ | 权重改变训练损失中类别贡献，不删除记录；欠采样把训练减少到 342+342 条，也改变缩放拟合样本。它们是完整训练策略对照。 |
| 为什么不能报告高 accuracy 就算成功？ | 正常交易占绝大多数，全预测正常也能有很高准确率；本项目必须同时报告 TP/FP、Recall/Precision@K 和 AP。 |
| 少标签实验有没有使用真实标签？ | 使用标签分层构造模拟掩码，明确披露。LR 只读取已标注记录；IF 读取全部无标签训练特征，因此不是只有算法不同的对照。 |
| 为什么标签实验的 IF 很弱，而主 IF 能检出 24 笔？ | E03 使用预先固定的 max_samples=256 参考，避免利用完整标签选出的改进配置；主对照用 IF4096。两者参数和研究目的不同，不能混用。 |
| 重要性说明 Amount 导致欺诈吗？ | 只能说明打乱 Amount 对固定模型的 AP 影响大；相关特征和模拟/匿名数据都限制解释，不是因果证据。 |
| 历史为什么降低 AP，却可能增加预算检出？ | 两个指标关注不同排序区域。当前受控结果保留这一区别，不能据此断言历史无用或为了好结果继续调参；两类历史仍是预定参考。 |
| 冷启动效果怎么样？ | 19 笔新客户交易没有欺诈，其他新实体组为空，无法估计这些组的欺诈检出能力；不重新生成数据凑正例。 |
| 85.96% recall 意味着模型很准确吗？ | 这是验证段的 49/57，审核了 568 笔，其中 519 笔正常，precision 只有 8.63%；概率也未校准，不能省略误报代价。 |
| 现在可以报告最终泛化结论吗？ | 尚无 A8 正式测试结果。验证已经用于选择和诊断，不能把验证优势当作独立测试结论。 |

## A8 交付后 B6/B7 的最终核对

1. 按 dataset_id、phase=test、数据/特征指纹和冻结版本核对输入；ULB 保持原训练政策，Handbook 明确区分 B 的初始静态模型与 A 的开发期滚动快照。
2. 对每个预定条件核对 N、欺诈数、K、TP+FP=K、Precision=TP/K、Recall=TP/F；冷启动全局审核分配和 TP 分配必须分别可加总。
3. 完整报告 E03 五档三种子及固定 IF256，不只选择表现最好的比例；E05 沿用既定置换协议和案例规则，不重新选特征或个案。
4. 正式图注、正文和海报明确写 test；保留验证材料的历史来源，解释负结果、零正类、空群体和金额覆盖分母的适用边界。
5. 先与公共汇总逐项对数，再补正式测试讨论与案例；完成后才勾选 B6/B7，并参加全组 G3 和演练。当前两项仍保留未完成状态。
