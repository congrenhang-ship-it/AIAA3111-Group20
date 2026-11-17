# B 分类方法与验证材料

整理：2026-11-17。本文保存分类方法、历史验证发现及讲解材料；正式测试已完成，结果以 [正式总览](../results/tables/project_test.json)、[ULB](../results/tables/ulb_test.json)、[Handbook](../results/tables/handbook_test.json)及 [标签与解释测试](../results/tables/b_extensions_test.json)为准。下文 validation 数字不能当作 test。

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

## B6 可排版英文正文：方法与验证结果

本节可由 D 分配至报告的 Methods、Validation Results 和 Limitations，而非独立按成员分章。预算表使用本稿已有的六模型对照来源和四视图表；详细条件保留补充材料。以下内容只报告验证结论。

### Classification and supervision protocol

We compare logistic regression (LR), a decision tree, Gaussian Naive Bayes and a random forest on the shared chronological ULB split. Each pipeline excludes the target and source-row identifier. StandardScaler is fitted within the training pipeline for LR and Gaussian Naive Bayes; the tree models use unscaled features. We additionally compare balanced class weights and seeded 1:1 random undersampling for LR. All minority training rows are retained during undersampling, with an equal-sized majority subset sampled without replacement. Validation retains its natural class distribution. The classifier is selected by validation true positives at the primary 1% review budget, followed by AP and a deterministic method-name tie-break. The capacity is K=ceil(Nq), with descending scores and ascending source-row identifiers resolving ties.

For the label-availability experiment, class-specific random permutations define nested 5%, 10%, 25%, 50% and 100% labeled subsets, using mask seeds 42/43/44 and a fixed model seed of 42. Each class count is rounded upward. The weighted LR pipeline is fitted only on the labeled subset. Stratification uses the true training labels and is therefore a controlled simulation, not an annotation-acquisition strategy. The fixed IF256 reference sees all unlabeled training features, so the comparison changes available records as well as supervision. Permutation importance uses validation AP, five repeats per original input and seed 42, without model refitting or subsequent feature removal.

### ULB validation findings

Weighted LR achieves AP 0.772681 and detects 49 of 57 frauds from 568 reviews (recall 85.96%, precision 8.63%). Original LR detects 48, and the random forest detects 47 despite its higher AP than original LR. Thus overall ranking quality and performance at a specified capacity can order methods differently. Undersampling also detects 48 frauds but reduces AP to 0.157422; it is not an overall improvement. The one-fraud gain of weighting should be interpreted against the small number of positives.

With 5% labeled training data, mean AP is 0.3799, ranging from 0.2562 to 0.6259 across the three masks; at 25%, mean AP is 0.7389. The full-label AP is 0.7727. These ranges describe sensitivity to the sampled labels and are not confidence intervals. The fixed IF256 reference has AP 0.016546 and recall 17.54% at 1%; it differs from the separately developed IF4096 used in the main anomaly comparison. Amount has the largest mean permutation AP decrease (0.580432), followed by V2 and V7. This describes model reliance, not causal effects or business meanings of anonymous features. Fixed diagnostic cases include high-scoring normal transactions and frauds below the review boundary; uncalibrated scores do not guarantee individual correctness.

### Handbook context and cold-start findings

We fit the same weighted LR separately on four A-provided feature views, using identical training and validation transactions and strictly past-only histories. Source data and partitions match A's handoff; B/C share the eight versioned v2 numeric view fingerprints. Transaction-only, customer-history, terminal-history and both-history AP values are respectively 0.255353, 0.218989, 0.235494 and 0.208312. At the common K=1,454, they detect 331, 348, 325 and 342 of 1,291 validation frauds. Histories therefore do not uniformly improve AP, although customer history increases detection at the declared capacity. Both-history remains the predeclared reference; the ablation does not trigger further view selection.

Cold-start groups are anchored to entities observed in the initial training segment. Of 145,315 validation transactions, 145,296 involve previously seen customers and terminals. Only 19 involve new customers, none are fraudulent, and the two new-terminal groups are empty. AP and recall are undefined for these groups; this dataset provides insufficient evidence about cold-start fraud detection. It cannot support demographic-fairness claims. All findings above concern validation, with formal held-out evaluation pending the group freeze. The simulator's designed mechanisms and the datasets' different prevalences also limit cross-dataset generalization.

### 正文图表与数值来源

| 正文内容 | 可用材料 | 限制 |
|---|---|---|
| ULB 六模型/策略对照 | [b_validation.json](../results/tables/b_validation.json)、[比较表](../results/tables/b_validation.csv) | 原始运行来源保留；统一入口已核对固定带权 LR |
| 标签曲线、置换前十项 | [组合图](../results/figures/b_extensions.png)、[完整汇总](../results/tables/b_extensions.json) | 三种子范围与五次置换标准差是不同统计量 |
| 历史消融、冷启动 | [组合图](../results/figures/handbook_b_ablation.png)、[完整汇总](../results/tables/handbook_b.json) | 四组静态模型，不与 A 的滚动快照混比 |
| 统一接入与共同环境验收 | [主线记录](../results/tables/a_team_validation.json) | 102 项通过为远程主线验收记录，不是本机新运行的检查 |

2026-11-16 整合更新：本稿的方法、标签/重要性及历史/冷启动验证正文已整合到 Report.md，Poster.md 已同步核心发现；旧的待交付和仅 ULB 范围描述已替换。讲解要点和 A8 后核对清单保留在本稿。技术测试及独立复现已完成；最终正文和海报整合仍见 TASKS.md。

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

## 报告与海报的最终核对

1. 按 dataset_id、phase=test、数据/特征指纹和冻结版本核对输入；ULB 保持原训练政策，Handbook 明确区分 B 的初始静态模型与 A 的开发期滚动快照。
2. 对每个预定条件核对 N、欺诈数、K、TP+FP=K、Precision=TP/K、Recall=TP/F；冷启动全局审核分配和 TP 分配必须分别可加总。
3. 完整报告 E03 五档三种子及固定 IF256，不只选择表现最好的比例；E05 沿用既定置换协议和案例规则，不重新选特征或个案。
4. 正式图注、正文和海报明确写 test；保留验证材料的历史来源，解释负结果、零正类、空群体和金额覆盖分母的适用边界。
5. 先与公共汇总逐项对数，再补正式测试讨论与案例；完成后才勾选 B6/B7，并参加全组 G3 和演练。当前两项仍保留未完成状态。


## 首次比较前固定的分类协议

### 候选与训练策略

1. 四种原始基线：逻辑回归、决策树、Gaussian NB、随机森林。
2. 只对逻辑回归比较 `class_weight='balanced'` 和一次训练集随机欠采样。
3. 欠采样保留全部少数类，从多数类无放回抽取同样数量，seed=42；保持抽取后的训练行原顺序。缩放器在该抽样训练部分拟合，验证集保持自然比例。此处比较的是完整训练策略。
4. 不额外扩大参数搜索。逻辑回归 max_iter=1000，随机森林 100 棵树；其余参数完整记录在结果 JSON 中。
5. 只使用 A 的原始数据校验、去重和时间划分。标签及 row_id 不进入特征。测试部分只核对分段身份，不打分。

逻辑回归和 Gaussian NB 用 StandardScaler，树模型不缩放。所有方法输出按验证行序排列的 Class=1 概率。

### 验证选择

沿用 C 的主要审核预算 1%，并报告 0.1%、0.5%、1%、2%。复用 `evaluate_scores`，K=ceil(N×q)，按 score 降序、row_id 升序稳定排名。选择验证集 1% 预算下 TP 最大的配置；并列时 AP 较高者优先，仍并列按方法名称升序。

操作决策直接使用 top-K，固定概率阈值记为 null。只用概率阈值无法保证在分数并列时恰好审核 K 笔，因此不另选阈值来替代预算排名。模型仅在训练段拟合，冻结后不合并训练和验证段重训。

### 六方案原验证比较

| 方法 | AP | TP | FP | Recall@568 | Precision@568 | 训练秒数 |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression_original | 0.642429 | 48 | 520 | 84.21% | 8.45% | 0.58 |
| decision_tree_original | 0.301922 | 38 | 530 | 66.67% | 6.69% | 11.34 |
| gaussian_nb_original | 0.033763 | 20 | 548 | 35.09% | 3.52% | 0.24 |
| random_forest_original | 0.757048 | 47 | 521 | 82.46% | 8.27% | 121.47 |
| logistic_regression_class_weight | 0.772681 | 49 | 519 | 85.96% | 8.63% | 0.94 |
| logistic_regression_random_undersample | 0.157422 | 48 | 520 | 84.21% | 8.45% | 0.05 |

按最大 TP、然后 AP、然后名称的既定规则，选中 **logistic_regression_class_weight**。这只代表限定候选在该验证窗口的选择，不能据此宣称测试表现或普遍优于其他算法。

类别权重版比原始逻辑回归多检出 1 笔欺诈，Recall 增加约 1.75 个百分点。随机森林的 AP 高于原始逻辑回归，但在主要预算下少检出 1 笔，说明模型选择必须遵守事先约定的审核任务。欠采样在该预算下与原始逻辑回归同为 48 笔，AP 明显下降，没有证据将它当作整体改进。

原始逻辑回归通过 A 的 `main.py --run-validation`，其 AP 与全部预算指标和独立 B 比较入口逐项一致。原始文件哈希和三段有序 row_id 指纹均与 A 的记录一致。选中模型保存后重新加载的验证风险分数在数值容差内一致，全部审核排序完全一致。

实际使用 Codex 辅助实现、合成检查、真实运行核对与材料整理；成员需核对并按实际使用情况完善 Disclosures。原逐次交接记录保存在 Git 历史中。
