# Data preparation

交接文档 / Handoff guide · [中文](../docs/TASK_A.zh-CN.md) · [English](../docs/TASK_A.en.md)

负责人 A：维护数据获取、版本、格式、校验值与划分说明。B/C/D 按此获取同一份数据。

## 每位组员独立准备数据

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

