# Task A 数据与实验交接

[English](TASK_A.en.md) · [任务清单](../TASKS.md) · [数据来源与预期指纹](../data/README.md)

A 为 Group 20 建立了统一的数据来源、清洗和时间划分，提供可复用的 Python 接口、G1 验证入口和相对时间窗口分析。沿用了前置实验的 ULB 原始版本、保留首条的去重口径和原始行号。Group 20 使用按行数约 60/20/20 的时间划分。前置审核回放使用固定 24/12/12 小时分段，因此两者的模型指标应各自注明实验协议。

## 文件结构

| 路径 | 状态 | 用途 |
|---|---|---|
| `code/download_data.py` | 新建 | 从固定 Zenodo 版本下载 CSV，校验后放入本地数据目录 |
| `code/preprocessing.py` | 完善 | 固定数据版本，读取和审计 CSV，统一去重、切分和提取特征 |
| `code/main.py` | 完善 | 审计、划分指纹和 G1 验证的命令入口 |
| `code/temporal.py` | 完善 | 窗口边界、各窗口交易和欺诈计数，调用 C 的公共评价 |
| `tests/test_task_a.py` | 新建 | 检查下载校验、固定行号指纹、时间窗与 B/C 接口 |
| `docs/TASK_A.zh-CN.md`、`docs/TASK_A.en.md` | 新建 | 中英文交接、复现方法与验收范围 |
| `requirements.txt`、`README.md`、`data/README.md`、`TASKS.md` | 更新 | 固定依赖、运行说明、数据证据和验收进度 |

`data/creditcard.csv` 和 `.venv/` 由组员在本地生成，Git 会忽略它们。

## 从新克隆复现

1. 在仓库根目录用 Python 3.12 创建并激活虚拟环境。当前数据验证使用 CPython 3.12.14。

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell 用 `./.venv/Scripts/Activate.ps1`。macOS 或 Linux 用 `source .venv/bin/activate`。也可用虚拟环境内的 Python 解释器直接运行下列命令。

2. 下载数据、安装固定版本并检查结果。

   ```bash
   python code/download_data.py
   python -m pip install -r requirements.txt
   python -m pip check
   python code/main.py --check-data
   python code/main.py --prepare-data
   python -m unittest discover -s tests -v
   ```

   下载脚本只使用 Python 标准库。默认文件是 `data/creditcard.csv`，已有正确文件时直接校验并复用。文件错误时会退出并保留现有文件。可以先移走错误文件再重跑，也可以通过 `--output` 下载到另一位置，并给主入口传入 `--data` 指定该文件。网络较慢时可按[数据说明](../data/README.md)手动下载，再运行相同的校验。

3. 比较 CSV SHA-256 以及训练、验证、测试的三个 `row_ids_sha256`。预期值在[数据说明](../data/README.md)。完整数据测试应显示通过。显示跳过代表测试未找到默认路径的 CSV。组员交接时一并记录 `git rev-parse HEAD`、`git status --short`、`python --version` 和 `--prepare-data` 的输出。

## B、C、D 怎么接入

把示例放在 `code/` 下的脚本中，直接用 Python 运行该脚本。

```python
from preprocessing import DEFAULT_DATA_PATH, get_features, load_data, split_data

train, validation, test = split_data(load_data(DEFAULT_DATA_PATH))
X_train, y_train = get_features(train), train["Class"]
X_validation = get_features(validation)
```

- **数据与特征。** `split_data` 返回三张 Pandas DataFrame，保留原始 `row_id`、`Time`、`Class` 和特征列。`get_features` 按固定顺序取 Time、V1–V28、Amount。B 和 C 只在训练段拟合模型和预处理，D 只在训练段确定分箱和发现规则。测试段留到 G2 冻结后评分。
- **B → A。** `train_model(X_train, y_train, "logistic_regression", strategy="original", seed=42)` 返回拟合流程。`score_model(model, X_validation)` 返回与验证表行序一致的一维风险分数，越大越可疑。
- **C → A。** `make_prediction_frame(validation, scores, method)` 返回 `row_id, Class, Time, method, score`，保留验证段记录和行序。`evaluate_scores(y_true, scores, review_fractions, row_ids=..., threshold=...)` 提供总体和窗口公用的指标。CLI 输出用 JSON，评价结果使用普通字典、列表、标量和 `None`。
- **窗口分析。** `summarize_time_windows(predictions, window_seconds=3600, origin=0)` 返回计数和状态。`evaluate_time_windows` 另外委托 C 计算各非空窗口的指标。各方法使用同一批交易、同一窗口和相同预算比例。窗口左闭右开，空窗口的评价为 `None`。无正类的非空窗口由 C 将 AP 和 Recall 标为不适用。逐窗预算和全段 top-K 分开解释。
- **G1。** B2 和 C2 交付后，运行 `python code/main.py --run-validation`。当前入口只拟合训练段、评分验证段，预算比例为 0.1%、0.5%、1%、2%。等待模块时命令会指出依赖并退出。

## 对照 A1–A9

本地复现记录（2026-11-10）。在同一台 Windows 电脑的干净目录重新建立 Python 环境，通过本地 HTTP 传输完整 CSV 来验证下载、校验和重用流程。五项测试全部通过，两次运行 `--prepare-data` 的 JSON 逐字节一致，三个行号指纹也与独立 CSV 核对一致。固定依赖通过了逻辑回归、K-means、Isolation Forest、随机过采样和 Apriori 的小样例。这些样例验证运行环境。本次 Zenodo 远程传输中途停滞，完整远程下载留待组员独立复验。

| 项 | 当前状态和剩余验收 |
|---|---|
| A1 | 已固定源文件、校验值和依赖，本机环境已复验通过。另一位组员仍需独立下载并复核环境。 |
| A2 | 数据审计和清洗决定已在全量 CSV 上核对。 |
| A3 | 统一切分、特征入口和全量指纹已在本地核对。待按小组流程提交合并。 |
| A4 | 命令和 G1 调度已用样例验证。缺失 B2/C2 时会明确退出。 |
| A5 | 窗口边界、空窗口、无正类窗口及公共评价调用已用样例验证。真实窗口指标待 C2。 |
| A6 | 待 B2/C2 和第二台电脑上的 G1 验证结果。之后接入其余分类、异常和规则模块。 |
| A7 | 待 B5、C6、D4 和全组确认冻结配置、预算、窗口及代码版本。 |
| A8 | 待 G2 冻结后统一跑正式测试并核对结果证据。 |
| A9 | 数据与复现说明已准备。最终报告、时间分析、第二台电脑核对和提交包随正式结果完成。 |

## Handbook 和 PaySim（A-N1 / A-N2 / A-N3）

在仓库根目录用同一 Python 3.12 环境和固定依赖运行以下命令。输出目录须是新目录；完整数据、模型、阈值文件和逐行预测留在 data/。模型复跑时固定 BLAS 线程数为 1（Git Bash 用 export OPENBLAS_NUM_THREADS=1，PowerShell 用 $env:OPENBLAS_NUM_THREADS = "1"）。

~~~bash
python code/main.py --prepare-handbook --handbook-pilot --output data/handbook/pilot
python code/main.py --prepare-handbook --output data/handbook/v1
python code/main.py --run-handbook-c --output data/handbook/c_validation
python code/main.py --run-handbook-rolling --output data/handbook/e11_validation
curl -fL "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2" -o data/paysim_v2.zip
python code/main.py --run-paysim-validation --output data/paysim_validation
~~~

Windows PowerShell 中用 curl.exe。也可从 [作者的 Kaggle v2 发布页](https://www.kaggle.com/datasets/ealaxi/paysim1) 保存同一个 ZIP，或使用 --paysim-archive 指定路径；验证入口会核对 SHA-256。Handbook 入口自动下载并校验上游生成器，分发时只需要代码和文档。重复生成使用另一目录；后续命令用 --handbook-dir 指定该目录。

| 交付文件 | 用途 |
|---|---|
| code/handbook.py | 调用固定版本的 Handbook 公开生成器；生成 CSV 和 manifest，提供开发段加载函数 |
| code/temporal.py | 四套按 row_id 索引的历史特征、E14 重建接口以及 E11 滚动验证 |
| code/paysim.py | 审计并流式读取作者 ZIP，固定采样、时间切分和三组验证对照 |
| code/main.py | 数据准备与三条验证命令 |
| [handbook_a.json](../results/tables/handbook_a.json)、[E11 曲线](../results/figures/handbook_a_rolling.png)、[paysim_a.json](../results/tables/paysim_a.json) | 版本、分段指纹、快照与阈值、接口检查和验证聚合结果 |

**组员调用。** 从仓库根目录把 code/ 加入 PYTHONPATH 后调用：

~~~python
from handbook import load_handbook_development
from temporal import handbook_history_builder

data = load_handbook_development("data/handbook/v1")
train, validation = data["train"], data["validation"]
train_views, validation_views = data["train_views"], data["validation_views"]
~~~

- B 将同名视图按 train.row_id 或 validation.row_id 排齐后传给分类器，目标用对应的 Class。
- C 可直接用 --run-handbook-c。E14 用 handbook_history_builder(context, history_keep=...)，掩码只决定哪些行进入后续历史，每行仍保留当前特征。历史使用 [t−窗口, t)，同一时刻互不可见。
- D 用元数据的 row_id、Class 按行连接 both_history 的 Amount、customer_amount_ratio_7d、customer_count_7d、terminal_count_7d，再调用 mine_rules(..., dataset_id="handbook")。TX_FRAUD_SCENARIO 单独留给事后诊断。

**数据边界。** 60 天生成固定为 5,000 客户、10,000 终端、主种子 42，客户与终端生成器分别用 42、43，交易与欺诈事件沿用上游的客户号、日期种子。生成数据共 583,193 笔，训练/验证/测试各 292,139 / 145,315 / 145,739 笔。1 天试生成及两次生产生成均可由聚合记录核对。E11 更新日固定为第 30、37、44 天，最后一个验证日用相同的未拟合交易校准各自的 1% 阈值。PaySim 作者说明要求排除四个余额字段，E16 只使用请求类型和金额。正式留出评分由 A7/G2 冻结后在 A8 执行。
