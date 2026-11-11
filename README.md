# Credit Card Fraud Detection under a Limited Review Budget

AIAA 3111 Course Project · Group 20

负责人：A 维护运行与环境说明，D 协调文档整合，全组核对实验约定。

本仓库按课程最终提交要求组织，使用 ULB 信用卡交易数据研究有限审核预算下的欺诈检测。

## 项目范围与投入比重

主问题：在有限审核预算下，监督分类与无监督异常检测分别能发现哪些欺诈？它们在不同交易群体和相对时间窗口中的表现是否一致？模式挖掘提供可验证的特征组合分析。

以下百分比是组内工作投入建议，不是课程评分权重，也不代表算法组合分数。聚类、异常检测和时间分析纳入正式计划；LLM 与 Agent 不开展。

| 模块 | 建议投入 | 研究问题与最小实验 |
|---|---|---|
| 数据预处理 | 10% | 数据检查、统一时间划分、训练阶段缩放，建立可靠比较基础 |
| 模式挖掘 | 10% | 少量特征分箱后用 Apriori 发现短规则，比较两种分箱并验证覆盖与稳定性 |
| 分类 | 25% | 比较既定四种模型，对代表模型比较类别权重和一种重采样策略 |
| 聚类 | 15% | K-means 在不使用标签时发现哪些群体？用少量簇数设置分析稳定性及各簇欺诈覆盖 |
| 异常检测 | 25% | Isolation Forest 与聚类距离异常分数在相同预算下能检出多少欺诈？是否覆盖分类遗漏的样本？ |
| 时间分析 | 15% | 冻结模型后，不同相对时间窗口中的检出能力、误报和类别比例如何变化？ |

深度来自统一审核任务下的方法差异、互补性、分组和时间评价，不靠增加大量算法。所有模块的结论以实际实验为准。

## 统一实验路线

1. 先跑通一个分类基线及公共评价，再并行推进分类、K-means、Isolation Forest 和规则挖掘。
2. 分类比较逻辑回归、决策树、Gaussian Naive Bayes、随机森林；不平衡实验只用于代表模型，避免穷举。
3. K-means 只在训练特征上拟合，簇数通过训练部分的结构指标和稳定性检查选择，不能把两个簇直接当成正常与欺诈。轮廓系数使用固定子样本控制成本。训练后使用标签描述每簇样本量、欺诈比例、欺诈覆盖量，并报告各簇内的模型表现。
4. 异常方法在不使用标签筛选的混合训练数据上拟合。比较 Isolation Forest 和到最近训练簇中心的距离分数。高异常分不直接等同于欺诈；Isolation Forest 原始分数方向须转换为越大越可疑。
5. 使用同一验证集，比较分类、异常方法与随机审核基线的 AP、PR 曲线和预算表现。异常方法若使用验证标签选择参数或组合权重，应明确说明这一点；其训练仍不使用标签。
6. 在相同审核预算下报告两种方法名单中的重合欺诈、分类独有欺诈、异常方法独有欺诈。补充预先设定的合并名单策略：按验证阶段确定的配额从两种排序去重选入，直到总审核数恰好为 K。比较相同 K 下是否真正增加检出量，不直接比较容量翻倍后的并集。
7. 规则仅在训练阶段发现，验证后冻结。将规则覆盖情况与预测错误作关联分析，不把规则当成模型的因果解释或未知业务行为。
8. 参数、阈值、规则和组合策略在验证阶段冻结，再统一进行最终测试。测试结果不用于继续挑选方法。
9. 在最终测试部分按预先约定的相对时间窗口分析固定模型：报告交易数、欺诈数、AP、审核数量与检出能力。各方法使用相同窗口和预算口径。没有正类的窗口将 Recall、AP 标为不适用；不预设每个窗口都有足够欺诈。
10. 时间仅指相对经过时间，不解释为本地钟点、个人交易序列或长期季节性。两天数据只支持短期表现分析，差异也可能来自窗口样本量或类别比例，不直接认定为概念漂移。

### 报告与海报叙事

有限审核资源 → 分类基线 → 无标签群体与异常排序 → 相同预算下的互补与局限 → 群体和时间窗口中的可靠性 → 用规则和错误案例补充解释。

### 简化范围

只使用 ULB 数据。聚类先做 K-means，异常方法先做 Isolation Forest 和已有簇中心距离，不增加 DBSCAN、LOF、复杂神经网络或第二数据集。共同评价由同一模块生成。LLM、Agent 和额外前端不纳入计划。

## 目录

```text
report/                  报告编辑稿与最终 Report.pdf
poster/                  海报编辑稿与最终 Poster.pdf
code/main.py             命令入口
code/preprocessing.py    数据检查、划分与预处理
code/classification.py   分类模型与类别不平衡处理
code/unsupervised.py     C：K-means、Isolation Forest 与簇距离
code/evaluation.py       C：公共评价、等预算组合、群体分析与绘图
code/run_c_validation.py C：独立验证入口（直接调用 A 的固定划分）
code/c_reporting.py     C：聚合交付、配置记录与冻结后的测试评价
tests/test_c_modules.py  C：公共评价与无监督模块检查
code/temporal.py         A：相对时间窗口评价
code/pattern_mining.py   特征组合与关联规则挖掘
data/                    本地原始数据及下载说明
results/figures/         正式实验图表
results/tables/          正式实验汇总表
```

`report/C_HANDOFF.md` 提供 C 的运行和交接说明，`results/c_validation/` 已有真实验证证据；最终报告、海报和最终测试尚未完成。

## 环境安装

建议组员统一使用 Python 3.12。在仓库根目录执行：

```bash
python -m venv .venv
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖版本已固定，组员使用 Python 3.12 和同一份 `requirements.txt`。

## 数据准备

Task A 交接与文件索引 · [中文](docs/TASK_A.zh-CN.md) · [English](docs/TASK_A.en.md)

每位组员在仓库根目录运行 `python code/download_data.py` 获取并验证同一份原始数据。文件保存在被 Git 忽略的 `data/creditcard.csv`。手动下载、校验值和预期结果见 [数据说明](data/README.md)。

## 当前可运行的命令

在仓库根目录执行：

```bash
python code/main.py --help
python code/main.py --tasks
python code/main.py --check-data
python code/main.py --prepare-data
python -m unittest discover -s tests -v
```

`--tasks` 显示四人模块分工，不需要数据文件。

`--check-data` 输出 SHA-256、字段、标签、缺失、重复和时间/金额范围。`--prepare-data` 先验证文件 SHA-256，再报告去重后的三段样本、欺诈数、时间范围和有序行号指纹。测试用小样例检查接口，并在 CSV 存在时校对全量数据指纹。
也可以指定其他 CSV 路径：

```bash
python code/main.py --check-data --data /path/to/creditcard.csv
```

**A 的数据下载、审计、统一时间划分和时间窗口模块已接入；C 的公共评价、K-means、Isolation Forest、等预算组合、群体评价和绘图已实现。**

```bash
python -m unittest discover -s tests -v
python code/run_c_validation.py --demo --output data/c_demo_new
python code/run_c_validation.py --data data/creditcard.csv --output data/c_validation_new --publish results/c_validation_new
```

合成演示输出到本地 `data/c_demo/`，仅验证程序。C 已完成真实训练/验证运行，汇总与 7 张图保存在 `results/c_validation/`，分析见 [C 验证结果](report/C_VALIDATION_RESULTS.md)。运行和后续交接见 [C 模块交付说明](report/C_HANDOFF.md)。G1 入口为 `python code/main.py --run-validation`，调用 B 的分类和 C 的评价；B2 尚未实现时明确报告依赖。正式测试安排在 G2 冻结后；不要把合成演示当作真实实验结果。

## 实施顺序与时间安排

当前安排从 2026 年 11 月 9 日开始，按现有项目材料中的海报和最终提交日期规划；如 Canvas 更新，以最新通知为准。

| 时间 | 工作与验收 |
|---|---|
| 11 月 9–11 日 | 统一数据、划分、评价接口；在第二位成员电脑上复现逻辑回归验证结果 |
| 11 月 12–16 日 | 并行完成分类比较、K-means、Isolation Forest、簇中心距离和小规模 Apriori |
| 11 月 17–20 日 | 验证阶段完成分群、预算和互补性实验，确定时间窗口规则，冻结参数与规则 |
| 11 月 21–22 日 | 统一最终测试，生成窗口表现、误报漏报、规则和群体分析 |
| 11 月 23–26 日 | 完成报告初稿、核心图表和海报，全组核对证据与结论 |
| 11 月 27 日 | 提交 Poster.pdf |
| 11 月 28 日–12 月 4 日 | 复现核查、问答演练和 Poster Session，不根据测试结果重选最优模型 |
| 12 月 5–11 日 | 完善局限、运行说明、报告与提交包，12 月 11 日最终提交 |

## 四人分工

逐项执行与完成标准见 [TASKS.md](TASKS.md)。每个人沿自己的编号连续推进，公共交付只集中在基础接口、方案冻结和最终证据核对三个节点。

以下 A/B/C/D 由组员自行对应姓名。每个公共代码文件由一人主责，其他成员通过 PR 提议修改。

| 成员 | 职责 | 主要文件 | 交付物 |
|---|---|---|---|
| A：数据、时间分析与整合 | 数据获取检查、统一划分、相对时间窗口分析、运行整合 | `code/preprocessing.py`、`code/main.py`、`code/temporal.py`、`data/README.md` | 统一数据、可复现入口、各窗口统计和表现分析 |
| B：分类模型 | 四种基础模型、有限调参、不平衡处理，提供分类风险排序 | `code/classification.py` | 模型流程、验证风险分数、冻结后的分类方案 |
| C：聚类、异常检测与统一评价 | K-means、Isolation Forest、簇距离、统一预算评价及互补分析 | `code/unsupervised.py`、`code/evaluation.py`、`results/` | 聚类画像、异常排序、共同评价和等预算组合结果 |
| D：模式挖掘与文档协调 | 分箱、Apriori、规则验证，协助跨方法案例分析和文档整合 | `code/pattern_mining.py`、`report/`、`poster/` | 规则结果、代表案例、统一排版的报告和海报 |

C 的范围限于既定 K-means、Isolation Forest、簇距离与公共评价，不额外增加算法。A、B、D 各自整理本模块图表和分析，避免将全部结果分析集中给 C 或全部写作集中给 D。

### 共同接口

- A 提供唯一的数据划分。组员先运行 `--prepare-data` 确认数据一致，再从各自位于 `code/` 的 Python 脚本调用以下函数。
- B 的 `train_model` 返回含预处理的模型流程。B 与 C 均输出同一行序、同一交易编号对应的连续分数，统一为越大越可疑。聚类编号单独保留，不能直接作为欺诈预测标签。
- C 的 `evaluate_scores` 统一负责所有分类、异常与组合方法的指标。提前约定 K 的取整、并列排序、无正类窗口及零分母处理。A 的时间分析直接复用这一接口。
- D 只在训练阶段拟合分箱并发现规则，在验证集检查稳定性，冻结选择后再进行测试评价。
- 标准化、特征选择、重采样和规则发现均遵守训练边界。参数、阈值和规则筛选在验证阶段确定。

```python
from preprocessing import DEFAULT_DATA_PATH, get_features, load_data, split_data

train, validation, test = split_data(load_data(DEFAULT_DATA_PATH))
X_train, y_train = get_features(train), train["Class"]
X_validation = get_features(validation)
```

`load_data` 返回附有原始 `row_id` 的数据表。`split_data` 负责去重和唯一时间划分，三段都保留标签、时间和原始行号。`get_features` 返回 Group 20 声明的 Time、V1–V28、Amount，不包含 `Class` 或 `row_id`，也不进行缩放。若要比较是否纳入 Time，先在验证阶段决定并记录。D 的规则挖掘从训练表读取特征和后件标签，并复用验证表的原始行号。

### 公共预测表

分类与异常方法统一使用以下列，表格保留原始交易行身份，不把 row_id 当成模型输入：

| 列 | 含义 |
|---|---|
| `row_id` | 原始 CSV 的稳定行编号，排序/划分后保持不变 |
| `Class` | 真实标签，仅用于评价 |
| `Time` | 相对经过秒数，用于窗口分析 |
| `method` | 模型与策略的唯一名称 |
| `score` | 连续风险分数，越高越可疑 |
| `cluster_id`（可选） | 群体编号，不等同于欺诈标签 |

B/C 的打分数组与输入行顺序一致，C 的 `make_prediction_frame` 负责统一组表。`evaluate_scores` 必须接收 `row_ids`，用于确定并列排序；A 的 `evaluate_time_windows` 复用该函数。按群体或窗口单独分配预算与全局 top-K 的分布是不同分析，结果必须注明口径。无正类时 AP/Recall 为不适用；空窗口另行标记。D 的规则结果保留覆盖的 row_id，便于交叉分析。

`evaluate_time_windows` 按共用起点划分左闭右开的窗口，并检查各方法使用同一批交易。返回各方法从首个到末个观测窗口的编号、边界、交易数、欺诈数、状态和 C 的评价结果。空窗口的评价为 `None`。在 C 的评价就绪前，`summarize_time_windows` 可独立生成同样的窗口计数。当前验证入口用 3,600 秒窗口和起点 0，正式方案在验证阶段确定。

每个 Python 文件开头均写明负责人、输入输出和实现状态。`NotImplementedError` 表示待实现，不能将接口数量当成完成实验的数量。

### 第一个共同里程碑

A 接好数据检查和划分，B 提供一个逻辑回归基线，C 生成第一张验证集审核预算表，由 B 或 C 在另一台电脑通过统一入口复现；D 后续在最终材料整合前再复核一次，避免早期流程等待规则模块。达成后并行扩展分类、聚类、异常检测和模式挖掘；A 准备时间窗口分析，D 在此之前可准备规则编码和报告提纲。

### 报告与海报

A 撰写数据、预处理和时间分析，B 撰写分类方法及实验，C 撰写聚类、异常检测、统一预算与互补分析，D 撰写模式挖掘并整合排版。Introduction、Conclusion 和 Limitations 由全组讨论后共同完成，所有成员都应能解释完整主线。篇幅重点放在等预算比较、互补性、群体差异和时间结果，而不是每种算法各写一节独立故事。

## 协作约定

- 各自在任务分支修改，通过 Pull Request 合并到 `main`。
- 修改公共函数的输入或输出前，先与其他成员确认。
- 仅提交代码、说明、编辑稿和选定的正式结果；不提交原始数据、虚拟环境、缓存或临时模型。
- 匿名特征只按 V1-V28 等原名报告，不虚构业务含义。
- 报告记录实际使用的外部协作和 AIGC 辅助。

## 提交整理

报告正文 6–10 页，含图表；参考文献、Disclosures、附录不计入正文页数。
报告编辑稿可保存为 `report/Report.docx`，最终导出 `Report.pdf`。
海报编辑稿可保存为 `poster/Poster.pptx`，最终导出 `Poster.pdf` 并单独提交。

最终 ZIP 建议整理为以下内容，并使用实际 Canvas 组编号命名：

```text
Group_20_Project.zip
├── Report.pdf
├── README.md
├── requirements.txt
├── code/
├── data/
│   └── README.md
└── results/
    ├── figures/
    └── tables/
```

`results/` 为补充结果。打包时应检查 README 中的路径与运行说明仍然适用于提交包。

