# Credit Card Fraud Detection under a Limited Review Budget

AIAA 3111 Course Project · Group 20

更新：2026-11-15。研究有限审核容量下，分类、异常检测与规则解释的能力、上下文价值和可靠性。保留 ULB 主实验，正式新增 Handbook 交易模拟数据及上下文/时间实验；PaySim 用于轻量外部复核。A 已交付 E09 数据和接口、E11 滚动验证及 E16 审计和验证。各线正式测试在 G2 后执行。不开展 LLM、GNN、复杂神经网络或前端。

负责人：A 维护公共入口和环境，D 整合材料；每位成员负责自己的实现、分析和来源记录。

组员建议先读 [项目故事与逻辑线](report/PROJECT_STORY.md)，再看 [剩余任务](TASKS.md) 和 [扩展协议](report/EXTENSION_PROPOSALS.md)。

## 当前进度与分支

`Task_A` 已整合 A/B/C/D，在全量 ULB 验证段复现共同模型和 E01/E02/E04/E06/E07，并产出 E08 固定阈值诊断。B 的 E03/E05 仍待交付。主线接入和全组 G2 确认后，按同一协议进入正式测试。

| 分支 | 已有交付 | 后续 |
|---|---|---|
| `Task_A` | ULB 统一验证、E08、Handbook 数据接口与 E11、PaySim 审计与验证 | 主线验收、接入 E03/E05、分线冻结和正式测试 |
| `b/classification-interface` | 四种分类器、不平衡对照、验证候选 | 分类扩展、最终结果与案例 |
| `codex/c-independent-modules` | 已接入 A 80f3171 及 D；统一验证、C 扩展和 E08 计数独立验收通过 | 正式图表与结论待 G2/A8 |
| `codex/d-pattern-mining` | 规则验证、E04 增量覆盖、共同环境复核、报告/海报内容稿 | 主线复现和正式材料 |

C 已提供接入说明和真实验证重放；D4 共同环境复核已有证据。已有实现不重新列成待开发任务，剩余工作见 [TASKS.md](TASKS.md)。C 已完成 A 统一入口的独立验收；后续合并保留最新分工和各人的真实状态。

统一入口、已交付候选与正式测试的验收状态见 [TASKS.md](TASKS.md)。

## 项目问题与课程联系

主线：有限审核资源 → 分类基线 → 聚类群体和异常排序 → 同预算互补 → 标签不足与可靠性 → 规则及错误解释 → 时间和阈值表现。

| 课程内容 | 项目对应 |
|---|---|
| 数据预处理 | 数据校验、去重、唯一时间划分、训练期缩放、稳定行身份 |
| Pattern Mining I/II | 分箱、Apriori、规则留出验证、重叠和增量覆盖 |
| Classification Basic/Advanced | 四种基线、不平衡处理、有限标签曲线、成本/阈值评价 |
| Clustering I/II | K-means、结构及稳定性、群体画像与表现 |
| Anomaly Detection | IF、簇距离、无标签参考、等预算组合 |
| Diverse Data | 新增客户/终端类别信息、严格历史统计、基础关系图特征和跨数据集验证；均为待实现任务 |

课程覆盖以实际实验和分析为准，不用增加算法数量代替深度，也不保证扩展会提高检测指标或成绩。

## 本轮正式新增：只用原数据

全部采用现有 ULB 数据，复用公共接口。详细默认设置、边界处理和验收见 [扩展实验协议](report/EXTENSION_PROPOSALS.md)。

C 已交付 E01/E02/E06/E07：原验证段 56,745 笔、57 笔欺诈；500＋200 轮配对分析复跑一致，代码/原模型及金额加总均核对，临时整合 D 后共 74 项检查通过。结果与接入说明位于 C 分支：[汇总](https://github.com/congrenhang-ship-it/AIAA3111-Group20/blob/codex/c-independent-modules/results/tables/c_extensions.json)、[C 说明](https://github.com/congrenhang-ship-it/AIAA3111-Group20/blob/codex/c-independent-modules/report/C_IMPROVEMENTS.md)。main 的业务代码尚待 A6 合入；上述均为验证结果，不是测试结果。

| 编号 | 内容与研究问题 | 主责 | 最小实验与结果 | 状态 |
|---|---|---|---|---|
| E01 | 检出笔数与欺诈金额覆盖是否一致？ | C | 原预算下金额覆盖、训练边界金额分组 | 验证交付完成 |
| E02 | 少量欺诈样本下差值是否稳定？ | C | 配对 bootstrap、时间块敏感性、AP/Recall 差值区间 | 验证交付完成 |
| E03 | 可见标签减少时分类与 IF 如何变化？ | B | 5/10/25/50/100% 标注、种子 42/43/44、固定参考 IF | 待验收 |
| E04 | 多条规则提供多少新增覆盖？ | D | Jaccard、固定顺序累计去重覆盖、分类/异常错误关联 | 待验收 |
| E05 | 模型依赖哪些特征，错误有何特点？ | B | 已选分类器 AP 置换重要性、固定案例规则 | 待验收 |
| E06 | 比简单审核参考强多少？ | C | 随机审核期望、Amount 降序、Precision 提升倍数 | 验证交付完成 |
| E07 | 预算如何影响不同错误代价？ | C | 四档预算，假设漏报/误报成本比 10/50/100 | 验证交付完成 |
| E08 | 固定分数阈值会产生多少后续审核量？ | A | 验证阈值，测试与固定窗口实际审核率和检出 | 验证交付完成 |

金额覆盖不称真实挽回损失；成本比是假设情景；重要性不作因果解释；规则不自动形成固定容量检测器。标签不足实验明确哪些标签/特征可见，IF 参考不利用完整标签新选参。保留负结果和没有组合收益的结果。

新增内容的工作量主要是有限对照和评价，不新增模型搜索。每位成员先做可独立完成的训练/验证工作；A6 整合可同时进行，A7 按数据集纳入对应新增协议再正式测试。

## 统一实验约定

1. 使用相同原始 CSV、SHA-256、去重及 A 的唯一时间划分。约 60/20/20，同一时间边界进入后段；row_id/Class 不作特征。
2. 缩放、模型、分箱及规则只拟合相应训练记录。标签模拟中的分类管线仅拟合已标注子集。验证用于已声明的选择与诊断，测试留到 G2 后。
3. 四种分类基线为逻辑回归、决策树、Gaussian NB、随机森林，保留原类别权重和重采样对照；不重新穷举算法。
4. K-means 群体不能当成欺诈类别，IF/簇距离分数越高越可疑，不当欺诈概率。C 已有训练内部选择与验证披露，A 合并时保持实际选择流程和说明一致。
5. 主要预算为 1%，补充为 0.1%、0.5%、2%。K=ceil(N*q)，score 降序、row_id 升序；名单组合去重补齐后仍恰好 K。
6. 同时报告 AP、审核计数、TP/FP、Precision/Recall、金额覆盖和随机参考；无正类或零分母标 N/A，空组/窗口单列。
7. 群体使用同一训练 K-means，分清全局 top-K 在群体的分配与独立组预算。固定阈值 score>=t 的实际审核数可能超过目标 K，单独报告。
8. 规则仅在训练发现、验证检查并冻结；后续覆盖、冗余及案例保留原规则来源。不能累加重叠规则的欺诈数。
9. 时间只解释相对经过时间；窗口起点、长度及独立窗口预算在 G2 中记录。两天数据的短时差异不证明长期漂移。
10. 配置、掩码种子、分箱/金额边界、规则顺序、阈值、成本比、重采样和解释协议在 A7 冻结。正式测试不回流选优；ULB 原训练边界不变，新数据滚动只发生在开发期；实现错误修复需记录影响和重跑原因。

新增 E03 的分层标注模拟使用真实训练标签构造子集，必须披露；不能隐藏这一信息后宣称完全模拟真实标注过程。已有验证已被查看，也应在方法和局限中说明。

## 新增数据集已进入正式任务：深度来自受控对照

2026-11-15 将原候选升级为任务 E09–E16，详细协议见 [扩展协议](report/EXTENSION_PROPOSALS.md)，逐人步骤见 [TASKS.md](TASKS.md)。当前仅完成设计和分工，未生成新数据或运行新实验。

主问题扩展为：**在相同审核容量下，历史与关系上下文带来多少价值，这些收益在标签延迟、冷启动和历史不完整时是否可靠？**

| 编号 | 正式内容 | 负责 | 主要对照与产出 |
|---|---|---|---|
| E09 | Handbook 数据与无泄漏历史接口 | A | 可复现生成、时间划分、单笔/客户/终端历史视图 |
| E10 | 历史特征消融 | B 分类、C 异常 | 单笔、加客户、加终端、两类历史；同一 LR/IF 与预算 |
| E11 | 开发期滚动验证与标签延迟 | A，复用 B 管线 | 固定模型 vs 每 7 天更新；延迟 0/1/7 天，最终测试前锁定快照 |
| E12 | 基础客户–终端关系特征 | C | 历史特征上增加过去度数/新边标记，不用 GNN |
| E13 | 冷启动分群 | B，复用 C 评价 | 初始训练已见/新客户、新终端，检查全局名单分配 |
| E14 | 历史缺失压力测试 | C | 隐去 0/10/30% 历史记录，当前交易与标签不变 |
| E15 | 有业务含义的规则及机制诊断 | D | 训练发现短规则、留出验证、按生成机制解释成功/失败 |
| E16 | PaySim 可行性与轻量外部复核 | A | 先审计来源/字段/资源；通过后仅 Amount/LR/IF 小对照 |

Handbook 是本轮主要第二数据集，使用[公开生成方法](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html)及[历史特征参考](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineFeatureTransformation.html)。采用 60 天、seed=42，先完成一天试运行，生产规模和生成设置在建模前固定。它有人工设计的欺诈机制，结果用于受控研究，不能宣称现实泛化。

A 已通过 [PaySim](https://github.com/EdgarLopezPhD/PaySim) 来源与字段审计，使用作者的 v2 ZIP 完成 E16 三组验证。来源、固定采样与结果见 [E16 聚合证据](results/tables/paysim_a.json)。正式测试按 A7/A8 处理。

ULB 保留原切分、特征和已完成成果；不同数据集分别训练、分别评价，不拼表，不直接套用 ULB 模型。AP 随类别比例变化，跨数据集主要比较方法的相对表现与局限，不平均成一个总分。所有结果带 dataset_id。

A 在完整 Handbook 数据上运行了 C 的正式验证入口，复跑后历史消融、关系增量、九个压力条件和视图指纹一致。另用 B 分类器和 D 规则模块完成真实接口检查。各成员的分析与发布状态见 [TASKS.md](TASKS.md)，A 的证据见 [Handbook 聚合记录](results/tables/handbook_a.json)。

数据生成、下载、B/C/D 调用与文件结构见 [中文交接](docs/TASK_A.zh-CN.md) / [English handoff](docs/TASK_A.en.md)。

深度优先级：先完成 E09/E10/E15 的第二数据集闭环，再补 E11/E12，之后 E13/E14 压力分析；PaySim 最后。E13 的群体若样本/欺诈不足就如实报告，不反复生成直到出现好看的结果。新增方向均已分派任务，优先级不等于已完成。

## 冻结与报告安排

继续使用 G2/G3 两个公共节点，但 G2 按数据集分别记录配置与验收状态。ULB 已具备条件即可冻结/测试，不等待 PaySim 或尚未就绪的新数据线；Handbook 准备好后冻结自己的协议，再评分自己的测试段。当前尚未存在已确认的 G2/A8。

历史特征只读取当前时刻之前的交易，同时间批次预测后再更新状态。标签延迟是明确的模拟假设，不能使用尚未到达的标签。新数据的开发期滚动可以使用当时已可得的历史开发标签；进入最终测试前保存各情景的固定快照，测试期不训练、不用测试标签回填。真正测试期在线更新暂不开展。[时间验证参考](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)

正文仍为 6–10 页：ULB 作真实匿名数据主线，Handbook 用 1–2 张组合图解释上下文与可靠性；完整压力条件、规则和 PaySim 放补充材料。高分目标落实为公平比较、可解释发现、可复现闭环和答辩证据，不承诺增加数据或模块就必然提高成绩。

## 简单目录与文件责任

```text
README.md                         公共路线、状态与运行说明（A）
TASKS.md                          剩余分工（D 维护，各人更新自己的任务）
code/main.py                      公共调度与冻结/最终运行（A）
code/preprocessing.py              数据适配、数据版本和各自唯一划分（A）
code/temporal.py                   历史特征、滚动/延迟、时间与阈值诊断（A）
code/classification.py             分类、标签预算与解释（B）
code/unsupervised.py               K-means、IF、基础图特征与压力实验（C）
code/evaluation.py                 公共评价、金额、区间、参考、成本、发布（C）
code/pattern_mining.py             规则和覆盖分析（D）
report/EXTENSION_PROPOSALS.md       E01–E16 协议、状态和条件（全组遵守）
report/                           各模块已有说明、报告及 Report.pdf
poster/                           海报稿及 Poster.pdf
results/tables/                    汇总 JSON/CSV；不逐实验新建目录
results/figures/                   精选图；注明 validation/test
data/                            本地数据与下载说明（原始文件不提交）
```

沿用各分支已有运行脚本，扩展放回主责模块；不为每项扩展新增脚本、Notebook 或层级。完整预测、模型、样例行和临时导出保留被忽略的本地 data/。细节见扩展协议的汇总约定。

## 环境、数据与可用命令

在含 A6 的分支上使用 Python 3.12，安装仓库固定依赖。从仓库根目录运行以下命令。

```bash
python -m pip install -r requirements.txt
python code/download_data.py
python code/main.py --prepare-data
python code/main.py --run-team-validation --output data/team_validation
```

数据来源、原始校验值和预期划分见 [数据说明](data/README.md)。输出目录需位于 `data/` 且尚未存在。统一入口使用已交付的配置，仅在训练段重建三个模型，复用同一批验证预测运行总体、分群、时间窗口、等预算组合、C 扩展、D 规则分析与 E08。验证通过后，模型、逐行预测和案例留在本地，汇总与图表位于输出目录的 `aggregate/`。E03/E05 在交付后接入此入口。

```bash
python -X utf8 -m unittest discover -s tests
python -X utf8 -m unittest discover -s code -p test_classification.py
```

本次验证阈值、逐窗审核计数和来源见 [A6/E08 验证摘要](https://github.com/congrenhang-ship-it/AIAA3111-Group20/blob/Task_A/results/tables/a_team_validation.json)。全组在 A7 确认配置后由 A8 运行正式测试。

## 分工与执行顺序

| 成员 | 现在可独立做 | 交付之后 |
|---|---|---|
| A | A-N1/N2/N3 的交付与验证已完成 | 各数据线具备条件即 A7/A8 |
| B | 先完成原 B-E1/B-E2；可用小样例准备新数据管线 | B-N1 历史对照 → B-N2 冷启动 → 原 B6/B7 |
| C | A 已运行 C-N1/N2/N3 正式验证入口 | C 复核并发布聚合结果；各线测试后 C7/C8 |
| D | 现有稿件继续；准备语义特征规则与机制解释 | A-N1 后 D-N1，逐线收取证据并做 D5–D7 |

具体编号、依赖和验收见 [TASKS.md](TASKS.md)。不等待他人最终图表再写已有方法；B/D 可先生成标准预测/覆盖，之后直接复用 C 的新增评价。C 不接管全部实验，D 不代写其他人的全部技术部分。

## 公共接口与结果

标准预测表为 `row_id, Class, Time, method, score`，可附 cluster_id；金额通过 row_id 从原始表对齐，不改变 B/C 现有表的必需列。完整行身份只在本地。

A 提供统一划分；B 的模型管线和 C 的冻结模型评分输出同一分数方向。C 的公共评价管理预算、并列、缺失指标、金额/成本和区间；D 提供按同一 row_id 的规则覆盖。A 时间评价复用 C 指标，避免独立实现另一种统计口径。

扩展阶段先发布 validation 汇总，G2 后以相同冻结方案输出 test 汇总。图表及每项结果应关联数据指纹、命令、配置、代码版本和 phase，拒绝覆盖已有正式证据。实验实现、运行结果和通过验收分别记录。

## 报告、海报与提交

报告正文保持 6–10 页（含图表），参考文献、Disclosures、附录按课程要求处理。新增内容使用组合图与简表：主问题/数据约 1 页，方法约 1.5 页，主对照与互补约 2 页，标签/可靠性约 1.5–2 页，规则/解释/时间约 1.5–2 页，结论和局限约 0.5 页；按实际内容压缩，总正文不超过 10 页。

全部重要性、种子结果、成本场景和抽样细节放补充表；正文优先清楚解释主要发现，不要求每个实验单独占页。海报围绕主要结果选图，不复刻全部报告。

A 写数据与时间/阈值，B 写分类/标签/重要性，C 写聚类/异常/公共评价与可靠性，D 写规则并整合。全组核对 Introduction、Conclusion、Limitations 和实际外部协作/AIGC 记录。没有改善的结果如实报告。

按依赖推进：A6 与扩展并行 → A7/G2 → A8/各人的正式分析 → G3 证据/第二台电脑复现 → PDF 导出、演练与提交。课程具体日期以 Canvas 最新要求为准。

最终目录沿用：report/Report.pdf、poster/Poster.pdf；ZIP 含 Report.pdf、README.md、requirements.txt、code/、数据说明及必要 results/，海报按要求单独提交。A9 核对包内路径、实际依赖和复现命令。

C 的后续验收：在完整验证段重现 A 的统一入口和 C/D 证据，81 项检查通过；新增固定阈值与窗口预算验证图，严格区分总审核 568 与逐窗取整后的 572。LR 阈值跨机器相差约 6.9e-11、计数一致，A7 应把模型与对应阈值一起冻结。源与解释见 C 分支说明及 c_extensions.json 的 team_integration；目前 GitHub 的 A 交付仍明确 E03/E05、G2 和 A8 待完成，验证图不作为正式测试图。
