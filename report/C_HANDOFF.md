# C 模块交付与运行说明

负责人 C。更新：2026-11-11。

**当前改进版本 v2 已完成：** 见 [C_IMPROVEMENTS.md](C_IMPROVEMENTS.md) 和 `results/c_improvements/`。训练内三种子选择 IF 抽样 4096，正式验证 1% 预算检出 24/57；43 项 A/C 检查通过。v1 的数据与下方说明保留作为原始基线；A8 使用 v2 时应调用 `c_improvements.evaluate_improved_frozen`。

已把 A 的 `Task_A`（`26c1c32`）合并到 C 分支，并处理 README/TASKS 的文字冲突；A 的数据/时间代码与 B 的分类代码没有被重写。

## 当前交付

| 任务 | 已完成 | 尚需的外部条件 |
|---|---|---|
| C1/C2 | 统一预测表、排名与预算口径、AP/PR、阈值计数、随机期望基线；人工例子与边界检查 | 无 |
| C3 | 真实训练段 K-means、k 对照、silhouette/ARI、训练与验证群体画像，含空簇 | 无 |
| C4 | 真实训练段 Isolation Forest、距离异常排序及相同验证预算比较 | 无 |
| C5 | 所有公共接口、真实分群评价、两种异常名单互补诊断、分类 CSV 接入与配额选择代码 | 分类与异常的正式互补结论需要 B5 |
| C6 | 无监督配置与管线已冻结保存；组合选择规则已固定 | B5 到位后选择分类组合；全组 G2 由 A 汇总 |
| C7 | 真实验证图表已生成；最终测试绘图和冻结评价入口已实现 | 最终测试图必须来自 G2 后的 A8 |
| C8 | 方法说明、真实验证分析、局限、讲解问答已整理 | B 的真实比较与 A8 最终测试解释需后续补入 |

真实结果见 [C_VALIDATION_RESULTS.md](C_VALIDATION_RESULTS.md)，预先记录的实验口径见 [C_PROTOCOL.md](C_PROTOCOL.md)，机器可读汇总与冻结配置见 `results/c_validation/`。没有把待 B5/A8 的实验勾成已完成。

## 文件分工

- `code/evaluation.py`：C 的公共指标、预测表、分群、名单组合及基础绘图。
- `code/unsupervised.py`：C 的 K-means、Isolation Forest、结构比较和冻结模型打分。
- `code/run_c_validation.py`：C 的独立验证入口，调用 A 的唯一划分。
- `code/c_reporting.py`：C 的聚合导出、结构/群体图、配置记录和 A8 冻结评价入口。
- `tests/test_c_modules.py`、`tests/test_c_workflow.py`：C 的指标、模型、A/C 联通和保存/恢复检查。
- `results/c_validation/`：C 的真实验证汇总和图表，允许随代码共享。
- `data/c_ulb_validation_20261111_final/`：本机完整预测、PR 点、模型及误报/漏报样例；Git 忽略，不共享原始数据或模型。

A/B/D 各自文件职责继续以 README 为准。

## 安装、验证、复现

在仓库根目录运行：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python code/download_data.py
python code/main.py --prepare-data
python -m unittest discover -s tests -v
```

本机为 CPython 3.12.14，已安装并通过 A 的完整固定依赖校验。核心版本 numpy 2.5.4、pandas 3.1.1、scikit-learn 1.9.1、matplotlib 3.11.2。35 项 A/C 检查通过，包括原始 ULB 文件、划分指纹、实际 C 评价与 A 的 G1/窗口接口联通。G1 联通测试使用模拟的 B 分数，并不等于 B2 的真实逻辑回归已完成。

正式 C 验证入口：

```bash
python code/run_c_validation.py \
  --data data/creditcard.csv \
  --output data/c_validation_reproduction \
  --publish results/c_validation_reproduction
```

输出目录必须新建或为空。`--data` 强制检查 A 的原始 SHA-256，直接调用 A 的 `split_data`，只给训练/验证段打分。已有 `results/c_validation/` 是此次真实验证证据，重现时使用上面的新目录，不覆盖旧记录。

`--publish` 只把聚合 JSON/CSV 与图导出到指定目录，和 GitHub 发布无关。完整预测、交易 ID、模型与错误案例保留在 `--output` 目录。汇总保留原始文件哈希、训练/验证有序 row_id 指纹、参数、代码 SHA-256、源 commit 和软件版本。

只做程序检查时：

```bash
python code/run_c_validation.py --demo --output data/c_demo_new
```

该输出明确标为合成数据，不能放入真实实验结果。

## B5 到位后，只需追加分类 CSV

B 对 A 的同一验证段生成 `row_id, Class, Time, method, score`。method 只有一个，名称使用英文字母、数字、下划线或连字符；连续 score 越大越可疑，不要传硬标签。B 的实际模型/训练设置、选优记录和 B5 来源也须交给 A。

```bash
python code/run_c_validation.py \
  --data data/creditcard.csv \
  --classification-predictions data/classifier_validation.csv \
  --output data/c_validation_with_B \
  --publish results/c_validation_with_B
```

行序可以不同，程序按 row_id 对齐，检查标签和 Time 一致，分群评价也保持同一组对应关系。先按 1% 预算下 TP、再按 AP 选择异常方法；本次无 B 时已选簇距离。分类配额候选 0/25/50/75/100%，在主要预算 TP 相同时保留更多分类容量。组合只挑一次，其他预算用于报告。保留纯分类基线，不强行认为混合更好。

CSV 也可由 A 导出后，用原先的 `--train ... --validation ...` 方式运行；它能校验时间顺序和行 ID，却不能单靠文件内容核实源数据身份，因此只有 `--data` 核验通过的记录自动标为固定 ULB 划分。

## A8 的最终测试接法（本次没有执行）

本机训练管线已保存为 `data/c_ulb_validation_20261111_final/c_models.joblib`。A 在自己的电脑重现上面的验证命令，就能得到自己的同配置模型，不需要传模型文件。仅加载自己可信的本地 artifact。

全组 G2 完成后，由 A 取得测试段并调用：

```python
import joblib
from c_reporting import evaluate_frozen_c

bundle = joblib.load("data/c_validation_reproduction/c_models.joblib")
# test 为 A 的唯一测试段；A 确认全组冻结后才执行。
result = evaluate_frozen_c(bundle, test, "data/c_final_test", phase="test")
```

此函数仅复用训练段拟合的管线和既定配额，不重新挑 k、异常方法或组合权重，不在 train+validation 上重新拟合。会生成完整预测、指标和正式图。B5 配额冻结后，A 可传 `classifier=classifier_test_predictions`；未冻结分类配额时拒绝做分类组合。无 B 输入时只输出 C 方法结果，不冒充完整小组最终比较。

保存/恢复检查可用 `phase="validation_replay"`；本次已在真实验证段做恢复核对。窗口分析由 A 的 `evaluate_time_windows` 对标准预测表继续处理。C 目前提供的一小时验证窗口仅作诊断，不替 A 选定最终时间窗口。

## 评价口径与解释边界

- K=ceil(Nq)，score 降序、row_id 升序处理并列；实际审核比例以 K/N 为准。AP 使用 sklearn 的非插值 average precision，同分按阈值组处理，不是梯形 PR-AUC。
- 无正类时 AP/Recall 为 N/A；空簇的样本数/K 为 0，比例、AP、Recall 不定义。训练有三个簇，本次验证簇 2 为空，图表与结果显式保留。
- 随机基线给 E[TP]=K×欺诈比例、E[Precision]=欺诈比例、E[Recall]=K/N；seed=42 的一次随机排序只是辅助，不把它当期望值。
- StandardScaler 只拟合训练段。无监督拟合不读取标签，也不先筛出正常交易。画像和性能评价使用事后标签；选异常方法/分类配额会用验证标签，必须披露。
- 群体编号不等于欺诈类别，异常分不等于欺诈概率，匿名 V 特征不赋予未知业务含义。
- 组内预算与一个全局 top-K 在各组的分配分别保存，不能把取整后组内容量之和当作同一个全局预算。
- `anomaly_comparisons` 复用公共名单函数：其中 classifier 字段表示第一个排序（Isolation Forest），anomaly 表示第二个（簇距离），实际没有分类模型参与。50/50 图只作诊断，不作为正式分类组合配置。
- 误报/漏报样例在本地 `private_error_examples.csv`，各方法最多各取五个高分误报和漏报。它们是代表性检查材料，不能用几个匿名样例证明因果或整体效果。

## 讲解准备

1. 为什么不是两个簇？正常交易也可能有多种结构，欺诈不一定集中在一个簇；k 来自训练结构比较。
2. 为什么 silhouette 与欺诈检出率可能不一致？前者评估特征空间紧致/分离，后者评估标签排序目标，不能互相代替。
3. 为什么保留表现较弱的 IF？这是既定方法对照；真实负面结果也是证据，不能验证后静默加算法换结论。
4. 为什么不直接比较两张 top-K 的并集？它的容量可能接近 2K；必须去重补齐后仍恰好 K 才公平。
5. 为什么组合不一定更好？异常独有欺诈可能伴随大量误报，并占掉另一个排序的容量；本次两种异常的 50/50 诊断并未优于纯距离。
6. 为什么不把这次结果当最终结论？它用于开发和选择，最终泛化要看全组冻结后的测试结果，且不能回头继续挑方案。

AIGC 记录：本次 Codex 辅助完成 A 分支合并、C 代码/检查、真实训练与验证运行、聚合导出和分析材料。组员应审核代码、数字和解释，并据实际使用纳入最终 Disclosures；本材料不是未经核对即可提交的最终报告。
