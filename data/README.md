# Data preparation

数据来源、下载、规模和划分。执行入口统一见 [README](../README.md)。

三套数据各自训练和评价，不混合记录或平均跨数据集 AP。

## 每位组员独立准备 ULB 数据

本组固定使用 [Zenodo 7395559 v1，2022-12-04](https://zenodo.org/records/7395559) 存档的 [creditcard.csv](https://zenodo.org/records/7395559/files/creditcard.csv)。原始数据来自 [ULB Machine Learning Group 与 Worldline](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)。原发布页列出 [ODbL](https://opendatacommons.org/licenses/odbl/1-0/) 和 [DbCL](https://opendatacommons.org/licenses/dbcl/1-0/)。

克隆仓库后，在根目录运行 `python code/download_data.py`。脚本使用 Python 标准库，把原始字节下载到 `data/creditcard.csv`，校验大小和 SHA-256 后才安装文件。若网络下载不顺畅，可从上面的直接链接手动下载，保持原始文件名和字节。已有同一文件时可直接复制。Git 已忽略该路径。

| 检查项 | 预期值 |
|---|---|
| 字节数 | 150,828,752 |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| MD5（Zenodo 发布值） | `e90efcb83d69faf99fcab8b0255024de` |

安装 `requirements.txt` 中的依赖后，从仓库根目录运行以下命令。`--prepare-data` 要求 SHA-256 匹配后才输出分段统计。审计、划分和测试都只读取原始 CSV。

```bash
python code/main.py --check-data
python code/main.py --prepare-data
python -m unittest discover -s tests -v
```

## 审计与处理依据

原始文件有 284,807 行、31 列，正常 284,315 笔、欺诈 492 笔，无缺失。Time 的范围为 0–172,792 秒，Amount 的范围为 0–25,691.16。Time 是相对秒数。V1–V28 是匿名 PCA 分量。Class 为 0（正常）或 1（欺诈）。

- 加载时按原始文件顺序赋予从 0 开始的 `row_id`，清洗和切分后继续使用原编号。
- 复用前置工作的去重口径。只合并包括 Time、Amount 和 Class 在内的 31 个原始字段均相同的记录，保留原 CSV 中第一条。移除 1,081 条额外重复行，其中正常 1,062 条、欺诈 19 条。这是可复现的记录级规则，匿名数据无法核实相同记录是否对应同一笔真实交易。
- 不因极端金额或欺诈标签删行。缺失值先在审计里报告，正式准备遇到必需字段缺失会停止，避免组员各自填补或删除。

## 共同时间划分

去重后按 `Time`、`row_id` 排序，取从 0 开始的第 `int(N × 0.60)` 和 `int(N × 0.80)` 个位置对应的时间戳为切点。切点上的全部记录进入后一段，同一时间戳不跨段。这里用 Group 20 的约 60/20/20 行数划分，前置审核回放中的固定 24/12/12 小时划分及其指标单独保存。

| 划分 | 实际最早–最晚时间（秒） | 笔数 | 欺诈数 |
|---|---:|---:|---:|
| 训练 | 0–120,392 | 170,235 | 342 |
| 验证 | 120,393–145,233 | 56,745 | 57 |
| 测试 | 145,234–172,792 | 56,746 | 74 |
| 合计 | 0–172,792 | 283,726 | 473 |

### 有序行号指纹 / Ordered row-ID fingerprints

`--prepare-data` 输出每段的 `row_ids_sha256`。计算时按划分后的顺序，将每个十进制 `row_id` 编为 ASCII，紧跟一个 LF（`\n`），再对所有字节求 SHA-256。文件 SHA-256 一致且三个指纹均匹配，表示三段的交易成员和顺序一致。

| Partition | Expected `row_ids_sha256` |
|---|---|
| Train | `1f69bf9bb0e7e5439a0896426eae142926d5e77b7d759a4e64bfa39c0a946333` |
| Validation | `2b3d3d2dd89b8a07a411d51d5d2afdf04b44922551bb7e6aad90aaef7a876de1` |
| Test | `4909103f5579d282c4faa1ca32c3c57fdf328e5d2ade9533a297d3e0254ca5c9` |

分段统计仅核对输入与标签计数。模型、分箱、缩放及重采样由各模块在训练阶段拟合。参数和方法在验证阶段确定，冻结后才评分测试集。


## Handbook

使用固定版本的公开模拟生成器，60 天、5,000 客户、10,000 终端，主种子 42（客户/终端种子 42/43）。生成器来源、版本、许可、配置和校验值见 [handbook_a.json](../results/tables/handbook_a.json)。

```bash
python code/main.py --prepare-handbook --output data/handbook/v1
```

入口自动下载并校验上游源码，生成 transactions.csv 和 manifest.json。重复运行使用新目录，后续命令传 --handbook-dir。训练/验证/测试为 30/15/15 天，共 583,193 笔：

| 划分 | 笔数 | 欺诈数 |
|---|---:|---:|
| 训练 | 292,139 | 1,752 |
| 验证 | 145,315 | 1,291 |
| 测试 | 145,739 | 1,280 |

历史只读取 [t−窗口, t)，同时间交易不可互见。实体 ID、标签及生成机制不作为模型输入。E11 开发期更新日为第 30、37、44 天，最终快照和阈值在测试前冻结；测试期不更新模型。模拟机制只用于事后解释，不能推断现实泛化。

## PaySim

使用[作者 Kaggle v2 ZIP](https://www.kaggle.com/datasets/ealaxi/paysim1)，保留 ZIP 原始字节：

```bash
curl --fail --location --retry 3 'https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2' -o data/paysim_v2.zip
```

Windows PowerShell 使用 curl.exe。也可从作者页面手动下载相同 ZIP，再用 --paysim-archive 指定路径。

- ZIP：186,385,561 字节。
- SHA-256：f7eef9ffad5cfa64a034143a5c9b30491d189420b273d5ad5723ca40b596613d。
- 流式读取完整文件后按固定、与标签无关的规则取样；不反复抽样选好结果。
- 排除作者说明不宜用于欺诈检测的余额字段，固定对照仅使用交易类型与金额。
- 完整抽样、切分、许可和指纹见 [paysim_a.json](../results/tables/paysim_a.json)；这是声明过的抽样实验，不声称用全部原始记录训练。

## 本地文件

原始数据、模型、逐行预测及运行日志不提交 Git。当前正式模型保存在 final_preparation_20261117/，正式预测在 final_test_20261117_r2/，首次中断及重跑证据保留在 final_test_20261117/ 和 final_test_failed_20261117/。这些本地文件不替代新克隆后的独立重建；重建命令见根 README。
