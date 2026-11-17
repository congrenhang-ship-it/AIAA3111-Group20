# Credit Card Fraud Detection under a Limited Review Budget

AIAA 3111 Course Project · Group 20

更新：2026-11-17。研究有限审核容量下，分类、异常检测与规则解释的能力、上下文价值和可靠性。ULB 为真实匿名数据主线，Handbook 用于上下文与时间机制实验，PaySim 用于轻量外部复核。E01–E16 已实现，三套数据的 G2 冻结、正式测试和测试图表已完成。本轮不修改报告和海报，不开展 LLM、GNN、复杂神经网络或前端。

组员建议先读 [项目故事与逻辑线](report/PROJECT_STORY.md)，再看 [剩余任务](TASKS.md) 和 [扩展协议](report/EXTENSION_PROPOSALS.md)。

## 当前进度与分支

全部成员分支已合入 main，成员分支保留。现在按技术任务统一收尾，不再等待原分工交接。

- [G2 冻结记录](results/tables/group_freeze.json)：在正式评分前记录模型、数据、参数、阈值和代码来源。
- [独立复现与技术核对](results/tables/reproduction.json)：实际运行环境、核心对照和随机森林差异记录。
- [正式测试总览](results/tables/project_test.json)：三套数据、176 组预算计数核对、17 张测试图及错误修复记录。
- [ULB](results/tables/ulb_test.json)、[Handbook](results/tables/handbook_test.json)、[PaySim](results/tables/paysim_test.json)：分别训练、分别评价的正式结果。
- [Handbook 规则](results/tables/handbook_d_test.json)、[分类标签/重要性](results/tables/b_extensions_test.json)、[金额/成本/区间](results/tables/c_extensions_test.json)及 [C 测试摘要](results/tables/c_test.md)。逐行预测、模型及错误案例保存在本地 data/。

| 保留的成员分支 | 本次合入的最新提交 | 交付 |
|---|---|---|
| Task_A | a5513f1 | ULB 统一验证/E08、Handbook 数据/E11、PaySim |
| b/classification-interface | b73df08 | 分类基线、E03/E05、Handbook 历史/冷启动，以及新增正文和讲解材料 |
| codex/c-independent-modules | 304baa3 | 聚类/异常及公共评价、C 全部扩展、指纹修复和 B/C 共同环境验收 |
| codex/d-pattern-mining | 33e07a3 | ULB 规则/E04、复现说明、报告/海报验证稿 |

上述分支无需重复拣选。E15、A7/G2、A8 和 C7 的技术图表均已完成；报告与海报仍保留原验证稿，需要之后单独整合正式结果和导出 PDF。正文页数要求仍在最终排版时核对。

查看最新主线：

```bash
git switch main
git pull --ff-only origin main
```

继续本人工作时，切回自己的成员分支后运行 git fetch origin 和 git merge origin/main，取得本次整合；无需删除或重建成员分支。

## 项目问题与课程联系

主线：有限审核资源 → 分类基线 → 聚类群体和异常排序 → 同预算互补 → 标签不足与可靠性 → 规则及错误解释 → 时间和阈值表现。

| 课程内容 | 项目对应 |
|---|---|
| 数据预处理 | 数据校验、去重、唯一时间划分、训练期缩放、稳定行身份 |
| Pattern Mining I/II | 分箱、Apriori、规则留出验证、重叠和增量覆盖 |
| Classification Basic/Advanced | 四种基线、不平衡处理、有限标签曲线、成本/阈值评价 |
| Clustering I/II | K-means、结构及稳定性、群体画像与表现 |
| Anomaly Detection | IF、簇距离、无标签参考、等预算组合 |
| Diverse Data | Handbook 历史、关系、标签延迟、缺失压力、机制规则及 PaySim 外部复核 |

课程覆盖以实际实验和分析为准，不用增加算法数量代替深度，也不保证扩展会提高检测指标或成绩。

## ULB 数据上的扩展（E01–E08）

全部采用现有 ULB 数据，复用公共接口。详细默认设置、边界处理和验收见 [扩展实验协议](report/EXTENSION_PROPOSALS.md)。

C 已交付 E01/E02/E06/E07：原验证段 56,745 笔、57 笔欺诈；500＋200 轮配对分析复跑一致。原验证结果与接入说明：[汇总](results/tables/c_extensions.json)、[C 说明](report/C_IMPROVEMENTS.md)。这些代码和材料现已合入 main；上述均为验证结果，不是测试结果。

| 编号 | 内容与研究问题 | 主责 | 最小实验与结果 | 状态 |
|---|---|---|---|---|
| E01 | 检出笔数与欺诈金额覆盖是否一致？ | C | 原预算下金额覆盖、训练边界金额分组 | 正式测试完成 |
| E02 | 少量欺诈样本下差值是否稳定？ | C | 配对 bootstrap、时间块敏感性、AP/Recall 差值区间 | 正式测试完成 |
| E03 | 可见标签减少时分类与 IF 如何变化？ | B | 5/10/25/50/100% 标注、种子 42/43/44、固定参考 IF | 正式测试完成 |
| E04 | 多条规则提供多少新增覆盖？ | D | Jaccard、固定顺序累计去重覆盖、分类/异常错误关联 | 正式测试完成 |
| E05 | 模型依赖哪些特征，错误有何特点？ | B | 已选分类器 AP 置换重要性、固定案例规则 | 正式测试完成 |
| E06 | 比简单审核参考强多少？ | C | 随机审核期望、Amount 降序、Precision 提升倍数 | 正式测试完成 |
| E07 | 预算如何影响不同错误代价？ | C | 四档预算，假设漏报/误报成本比 10/50/100 | 正式测试完成 |
| E08 | 固定分数阈值会产生多少后续审核量？ | A | 验证阈值，测试与固定窗口实际审核率和检出 | 正式测试完成 |

金额覆盖不称真实挽回损失；成本比是假设情景；重要性不作因果解释；规则不自动形成固定容量检测器。标签不足实验明确哪些标签/特征可见，IF 参考不利用完整标签新选参。保留负结果和没有组合收益的结果。

全部条件已按冻结方案完成正式测试，不使用测试结果重新选模型。

## 正式测试的主要发现

以下为固定 1% 审核容量下的结果；它们不是跨数据集综合排名。

| 数据集 / 固定方法 | AP | 检出欺诈 / 测试欺诈 | 审核量 |
|---|---:|---:|---:|
| ULB：预先选定的带权 LR | 0.7475 | 64 / 74 | 568 |
| ULB：IF4096 | 0.0660 | 41 / 74 | 568 |
| Handbook：单笔 LR | 0.2692 | 346 / 1,280 | 1,458 |
| Handbook：两类历史 LR | 0.2130 | 334 / 1,280 | 1,458 |
| PaySim：Amount 参考 | 0.0798 | 94 / 413 | 1,289 |
| PaySim：固定 LR | 0.0634 | 80 / 413 | 1,289 |
| PaySim：固定 IF | 0.0447 | 102 / 413 | 1,289 |

ULB 分类在固定审核容量下优于独立异常排序；互补、金额覆盖和区间仍需看完整对照。Handbook 加历史与关系未自动改善：两类历史加关系的 LR 检出 333 笔，低于单笔 LR 的 346 笔，应作为上下文收益依赖设置的证据。PaySim 的 AP 与固定预算检出顺序不同，说明单看 AP 不足以决定审核策略。模拟机制、短时间跨度、标签条件和假设成本均限制现实推广，保留全部负结果。

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

2026-11-15 将原候选升级为任务 E09–E16，详细协议见 [扩展协议](report/EXTENSION_PROPOSALS.md)，逐人步骤见 [TASKS.md](TASKS.md)。E09–E16 的开发验证及冻结测试均已完成。E15 从训练发现短规则，在验证段按固定规则选择分箱方案；最终保留 19 条等频规则，测试机制标签仅用于事后解释。

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

A 已通过 [PaySim](https://github.com/EdgarLopezPhD/PaySim) 来源与字段审计，使用作者的 v2 ZIP 完成 E16 三组验证。来源、固定采样与结果见 [E16 聚合证据](results/tables/paysim_a.json)。正式三组测试见 paysim_test.json。

ULB 保留原切分、特征和已完成成果；不同数据集分别训练、分别评价，不拼表，不直接套用 ULB 模型。AP 随类别比例变化，跨数据集主要比较方法的相对表现与局限，不平均成一个总分。所有结果带 dataset_id。

C 的历史/关系消融、九条件缺失压力和正式测试已完成，保留未改善的结果。跨平台视图指纹已修复：只在哈希时统一两列时间编码，模型输入不变。原验证材料见 [Handbook C](results/tables/handbook_c.json)，正式结果见 handbook_test.json。

数据生成、下载、B/C/D 调用与文件结构见 [中文交接](docs/TASK_A.zh-CN.md) / [English handoff](docs/TASK_A.en.md)。

深度优先级：先完成 E09/E10/E15 的第二数据集闭环，再补 E11/E12，之后 E13/E14 压力分析；PaySim 最后。E13 的群体若样本/欺诈不足就如实报告，不反复生成直到出现好看的结果。这些预定实验现均已实现并测试。

## 冻结与报告安排

G2 已于正式评分前提交。首次测试中 Handbook 冷启动汇总漏传预算参数，修复后使用同一批模型重跑；原模型哈希未变，ULB 的 26 份逐行预测与首次运行完全一致。冻结记录保留实现修订及中断记录，未调整模型、阈值、规则、数据或实验条件。

历史特征只读取当前时刻之前的交易，同时间批次预测后再更新状态。标签延迟是明确的模拟假设，不能使用尚未到达的标签。新数据的开发期滚动可以使用当时已可得的历史开发标签；进入最终测试前保存各情景的固定快照，测试期不训练、不用测试标签回填。真正测试期在线更新暂不开展。[时间验证参考](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html)

正文仍为 6–10 页：ULB 作真实匿名数据主线，Handbook 用 1–2 张组合图解释上下文与可靠性；完整压力条件、规则和 PaySim 放补充材料。高分目标落实为公平比较、可解释发现、可复现闭环和答辩证据，不承诺增加数据或模块就必然提高成绩。

## 简单目录

```text
README.md                         公共路线、状态与运行说明（A）
TASKS.md                          完成证据及剩余提交事项
code/main.py                      统一命令入口
code/finalize.py                  冻结、正式测试、独立复现及技术打包
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

沿用各分支已有运行脚本，扩展放回对应模块；不为每项扩展新增脚本、Notebook 或层级。完整预测、模型、样例行和临时导出保留被忽略的本地 data/。细节见扩展协议的汇总约定。

## 环境、数据与可用命令

在 main 上使用 Python 3.12 虚拟环境，安装仓库固定依赖。从仓库根目录运行以下命令。

```bash
python -m pip install -r requirements.txt
python code/download_data.py
python code/main.py --prepare-data
python code/main.py --run-team-validation --output data/team_validation
```

数据来源、原始校验值和预期划分见 [数据说明](data/README.md)。输出目录需位于 data/ 且尚未存在。统一入口在相同训练/验证划分上运行固定主模型、C 的总体/分群/时间/组合及可靠性、D 规则、E08，以及 B 的 E03/E05。E03 按五档比例和三个种子拟合带权 LR（相同完整子集复用拟合），另拟合固定 IF256；E05 对固定完整标签 LR 做 30×5 次置换，不新选参。

输出沿用原布局：aggregate/tables/team_validation.json 同时包含 E03/E05，aggregate/tables/b_extensions.json 保存 B 的完整汇总，aggregate/figures/b_extensions.png 为组合图；全部模型、15 份掩码和逐行错误案例留在 b_extensions/ 私有子目录。统一入口不再把 E03/E05 列为 pending。

```bash
python -X utf8 -m unittest discover -s tests
python -X utf8 -m unittest discover -s code -p test_classification.py
```

只单独复跑 B 的 ULB 标签/重要性扩展时，仍可使用原命令；与统一入口共用相同计算函数，无需日常重复运行两遍：

```bash
python code/run_b_validation.py --extensions --output data/b_extensions_main
```

Handbook 从同一数据接口运行各模块，以下命令用于新克隆后的新输出目录：

```bash
python code/main.py --prepare-handbook --output data/handbook/v1
python code/run_b_validation.py --handbook --output data/handbook/b_validation
python code/main.py --run-handbook-c --output data/handbook/c_validation
python code/main.py --run-handbook-rolling --output data/handbook/e11_validation
```

默认读取 data/handbook/v1；已有其他生成目录时传 --handbook-dir。PaySim 获取作者 v2 ZIP 后运行：

```bash
python code/main.py --run-paysim-validation --paysim-archive data/paysim_v2.zip --output data/paysim_validation
```

来源和获取步骤见 [数据说明](data/README.md)。以上均为开发/验证入口；已存在的运行目录应换新名字，以保留原证据。

验证阈值和逐窗统计见 [原验证摘要](results/tables/a_team_validation.json)；冻结阈值的正式表现包含在各 *_test.json 的 threshold_policies 中。

## 从头复现正式结果

使用 Python 3.12.14 和 requirements.txt 的锁定依赖。从全新克隆开始：

```bash
python -m pip install -r requirements.txt
python code/download_data.py
python code/main.py --prepare-handbook --output data/handbook/v1
curl --fail --location --retry 3 'https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2' -o data/paysim_v2.zip
python code/main.py --reproduce-final --output data/reproduction
```

该命令先按原训练/开发范围重建全部快照，核对预定协议，再评分测试并对照版本化汇总；无需原作者的本地模型。输出使用全新目录，data/reproduction/reproduction.json 为结果记录。默认读取 data/handbook/v1；其他路径传 --handbook-dir。自动执行入口见 [.github/workflows/reproduce.yml](.github/workflows/reproduce.yml)。

独立 Linux 重建已完成三套正式评分。原逐值检查发现补充随机森林基线的 AP 从 0.7845347 变为 0.7840486，八项金额组审核分配各差 1–5 笔；总体预算、TP/FP、召回和金额覆盖不变，其余正式结果（含阈值、时间窗及 C 群体/互补）通过相同数值精度的对照。复现记录逐项保留差异，状态会明确标为 core_checks_passed_with_platform_differences；不宣称随机森林逐值相同，也未因此修改模型或正式参考结果。

原始准备和评分入口为 --prepare-final 与 --run-final-test。仓库已有正式冻结记录，重建请使用上面的 --reproduce-final；它会把新冻结写入隔离目录，不覆盖原证据。

## 公共接口与结果

标准预测表为 `row_id, Class, Time, method, score`，可附 cluster_id；金额通过 row_id 从原始表对齐，不改变 B/C 现有表的必需列。完整行身份只在本地。

A 提供统一划分；B 的模型管线和 C 的冻结模型评分输出同一分数方向。C 的公共评价管理预算、并列、缺失指标、金额/成本和区间；D 提供按同一 row_id 的规则覆盖。A 时间评价复用 C 指标，避免独立实现另一种统计口径。

扩展阶段先发布 validation 汇总，G2 后以相同冻结方案输出 test 汇总。图表及每项结果应关联数据指纹、命令、配置、代码版本和 phase，拒绝覆盖已有正式证据。实验实现、运行结果和通过验收分别记录。

## 报告、海报与提交

报告正文保持 6–10 页（含图表），参考文献、Disclosures、附录按课程要求处理。新增内容使用组合图与简表：主问题/数据约 1 页，方法约 1.5 页，主对照与互补约 2 页，标签/可靠性约 1.5–2 页，规则/解释/时间约 1.5–2 页，结论和局限约 0.5 页；按实际内容压缩，总正文不超过 10 页。

全部重要性、种子结果、成本场景和抽样细节放补充表；正文优先清楚解释主要发现，不要求每个实验单独占页。海报围绕主要结果选图，不复刻全部报告。

本轮只完成实验、评价、图表、复现和技术包；报告/海报正文及排版未修改。后续写作须使用 *_test.json 和 *_test*.png 的正式结果，原 validation 材料作为开发过程证据。不同数据集的 AP 不合并平均；历史/关系未提升、模拟数据的局限和已查看过验证结果都需如实说明。

最终目录沿用 report/Report.pdf、poster/Poster.pdf；课程要求的 Group_20_Project.zip 还需包含最终 Report.pdf，海报按要求单独提交。本地 Group_20_Code.zip 是可运行的技术包，不是完整课程提交包。实际演练、成员确认、Canvas 提交及课程日期仍需全组自行核对。

生成技术包（原始数据/模型不入包）：

```bash
python -c "import sys; sys.path.insert(0, 'code'); from finalize import package_technical; print(package_technical('data/Group_20_Code.zip'))"
```

技术包保留必要的既有 C 操作说明和训练规则文件作为运行依赖；不含报告/海报稿。解压后依照 README 安装依赖、获取数据即可运行，不要求保留 Git 仓库。
