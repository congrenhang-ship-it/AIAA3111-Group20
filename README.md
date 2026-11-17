# Credit Card Fraud Detection under a Limited Review Budget

AIAA 3111 · Group 20

在有限人工审核容量下，比较分类、聚类/异常检测与规则解释，并研究标签不足、历史与关系信息、标签延迟、冷启动和缺失历史对结果的影响。ULB 是真实匿名交易主线，Handbook 用于受控上下文实验，PaySim 用于轻量外部复核。各数据集分别训练和评价，不平均跨数据集 AP。

E01–E16、测试前 G2 冻结、三套数据正式测试、17 张测试图和 Linux 独立重建均已完成。[报告 PDF](report/Report.pdf)已整合正式测试，正文 10 页，披露/参考文献/附录另计；[可编辑正文](report/Report.md)的成员姓名与学号留空待填写。海报暂未更新，剩余事项见 [TASKS.md](TASKS.md)。成员分支保留，日常使用 main。

## 目录

```text
README.md                 安装、数据、运行、结果及提交说明
TASKS.md                  技术完成记录和剩余提交事项
requirements.txt          固定依赖
code/                     核心算法、数据处理、评价和 main.py 入口
code/validation/          模块验证脚本、规则复核及分类检查
code/workflows/           下载、冻结、正式评分、独立复现及打包
tests/                   自动检查（分类检查另在 code/validation/test_classification.py）
data/README.md            数据来源、许可、下载、规模和划分
results/tables/            冻结配置、必要开发证据及正式结果
results/figures/           验证与正式测试图
report/                   Report.md、Report.pdf、构建脚本、方法与实验协议、必要规则
poster/Poster.md          海报草稿
.github/workflows/        Linux 独立重建与检查
```

算法代码集中在 classification.py、unsupervised.py、pattern_mining.py；preprocessing.py、handbook.py、paysim.py 和 temporal.py 负责数据及时间接口；evaluation.py 统一指标，main.py 提供统一命令。workflows/finalize.py 负责冻结、正式评分、独立复现和技术打包；validation/ 中的 run_* 与 check_d_freeze.py 用于开发验证及规则复核。共享选择规则和规则诊断保留在核心模块，算法不依赖验证脚本。

## 安装和获取数据

使用 Python **3.12.14**，在仓库根目录执行。Windows 激活命令为 `.venv\Scripts\activate`，PowerShell 下载用 `curl.exe`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python code/workflows/download_data.py
python code/main.py --prepare-handbook --output data/handbook/v1
curl --fail --location --retry 3 'https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2' -o data/paysim_v2.zip
```

来源、许可、手动下载、字节校验及划分见 [data/README.md](data/README.md)。Handbook 为固定模拟生成；PaySim 为预声明的固定抽样实验，不声称使用全部原始记录训练。原始数据、模型、完整预测和日志仅留本地 data/，不提交 Git。

## 完整复现

```bash
python code/main.py --reproduce-final --output data/reproduction
```

从原训练/开发范围重建全部快照，核对协议，再评分三个测试段并对照已发布汇总；无需作者本地模型。默认数据目录为 data/handbook/v1 和 data/paysim_v2.zip，其他路径分别传 --handbook-dir、--paysim-archive。输出目录须未存在；再次运行请换新目录。新证据写在隔离目录，不覆盖正式记录。

结果记录为 data/reproduction/reproduction.json。Linux 自动执行见 [工作流](.github/workflows/reproduce.yml)，实际证据见 [reproduction.json](results/tables/reproduction.json)。主要方法与扩展通过对照；补充随机森林的 AP 在 macOS/Linux 重建中相差约 0.00049，八项金额组审核分配各差 1–5 笔，总体预算、检出数和金额覆盖不变。程序逐项报告这些差异，状态为 core_checks_passed_with_platform_differences，不宣称完全逐值一致。

只验证 ULB 统一入口：

```bash
python code/main.py --run-team-validation --output data/team_validation
```

它包括分类/异常/规则、E01–E08、15 个标签条件及 30×5 次置换。其他模块命令可查看 `python code/main.py --help`。已有正式冻结，通常直接运行 --reproduce-final；--prepare-final 和 --run-final-test 是原冻结/评分入口，不用于覆盖已发布证据。

检查代码：

```bash
python -m unittest discover -s tests
python -m unittest discover -s code/validation -p test_classification.py
```

当前 98 项公共检查＋11 项分类检查通过。冻结配置、训练规则和部分 validation JSON 是实际重建输入，应保留；重复交接和过程记录从当前目录移除，原版本可在 Git 历史查看。

## 结果入口

| 内容 | 文件 |
|---|---|
| 项目故事与研究问题 | [PROJECT_STORY.md](report/PROJECT_STORY.md) |
| E01–E16 参数及边界 | [EXTENSION_PROPOSALS.md](report/EXTENSION_PROPOSALS.md) |
| 正式测试前的模型/数据/协议冻结 | [group_freeze.json](results/tables/group_freeze.json) |
| 正式测试总览与一致性核对 | [project_test.json](results/tables/project_test.json) |
| 三数据集正式结果 | [ULB](results/tables/ulb_test.json)、[Handbook](results/tables/handbook_test.json)、[PaySim](results/tables/paysim_test.json) |
| 标签条件与重要性 | [b_extensions_test.json](results/tables/b_extensions_test.json) |
| 金额、成本和区间 | [c_extensions_test.json](results/tables/c_extensions_test.json) |
| C 群体/互补与写作数字 | [c_test.json](results/tables/c_test.json)、[c_test.md](results/tables/c_test.md) |
| 语义规则及机制诊断 | [handbook_d_test.json](results/tables/handbook_d_test.json) |
| 独立重建与技术包验证 | [reproduction.json](results/tables/reproduction.json) |

正式图在 results/figures/，文件名包含 test；其他图保留 validation 或训练结构含义。不要将验证数字改名为测试结果。

固定 1% 审核容量下的主要结果：

| 数据集 / 固定方法 | AP | 检出 / 测试欺诈 | 审核量 |
|---|---:|---:|---:|
| ULB：预选带权 LR | 0.7475 | 64 / 74 | 568 |
| ULB：IF4096 | 0.0660 | 41 / 74 | 568 |
| Handbook：单笔 LR | 0.2692 | 346 / 1,280 | 1,458 |
| Handbook：两类历史 LR | 0.2130 | 334 / 1,280 | 1,458 |
| PaySim：金额参考 | 0.0798 | 94 / 413 | 1,289 |
| PaySim：固定 LR | 0.0634 | 80 / 413 | 1,289 |
| PaySim：固定 IF | 0.0447 | 102 / 413 | 1,289 |

历史与关系没有自动改善，AP 与固定预算检出的排名也不必相同。保留负结果；模拟机制、短时间跨度、少量欺诈及假设成本限制现实推广。

## 实验约定

- 使用固定原始字节和唯一时间划分；缩放、分箱和模型只在合法训练范围拟合。历史只见过去，同时间交易互不可见；机制标签不入模。
- 主预算 1%，另报 0.1%、0.5%、2%。K=ceil(N×q)，score 降序、row_id 升序；名单组合去重补齐后仍为 K。
- 固定阈值与 top-K 分开评价；全局名单的群体分配与各群体独立预算分开解释。空组/无正类保留 N/A。
- 测试不回流调参。首次评分的冷启动汇总错误修复后用同一批模型重跑；原模型哈希和 ULB 26 份预测不变，修订已记入 G2。
- 金额覆盖不等于挽回损失，异常不等于欺诈，规则覆盖不等于有限容量收益，重要性不作因果解释。

## 报告更新与构建

报告正文采用常规 11pt Times 风格字体，5 张矢量图从正式 JSON 重建；10 张表覆盖主要对照、可靠性与机制诊断。正文 10 页，含披露、参考文献及附录的 PDF 共 13 页。

编辑 report/Report.md 后，使用项目 Python 环境及另行安装的 Pandoc、XeLaTeX 重新生成：

```bash
python report/build_report.py
```

脚本只绘图和排版，不训练模型或修改正式结果；会检查正文页数是否在 6–10 页。PDF 图保存在 results/figures/report_*.pdf，PNG 预览和 LaTeX 临时文件留在忽略的 data/report_build/。常规实验运行不需要 Pandoc 或 XeLaTeX。

## 课程提交

课程要求最终提交 **Group_20_Project.zip**，包含 Report.pdf、代码、README 和数据或下载说明；Poster.pdf 单独提交，不要求放入最终 ZIP。报告正文 6–10 页，图表计入；参考文献、Disclosures 和附录不计入正文。组员需核对内容并如实披露外部协作及 AIGC 使用。

本地技术包可由以下命令生成。它已通过无 Git 环境的完整 ULB 验证，但缺少最终 Report.pdf，因此不是完整课程提交包。

```bash
python -c "import sys; sys.path.insert(0, 'code'); from workflows.finalize import package_technical; print(package_technical('data/Group_20_Code.zip'))"
```

技术包不含原始数据、模型、逐行预测或报告/海报；保留必要训练规则及 C 操作说明作为运行依赖。报告已生成，填写成员信息并核对后加入最终课程包；海报、演练和 Canvas 提交仍见 TASKS.md。
