# Credit Card Fraud Detection under a Limited Review Budget

AIAA 3111 Course Project · Group 20

本仓库按课程最终提交要求组织，使用 ULB 信用卡交易数据研究有限审核预算下的欺诈检测。

## 项目范围

1. 第一优先级：分类模型比较、类别不平衡处理、审核预算评价和错误分析。
2. 第二优先级：Apriori 特征组合与关联规则挖掘。
3. 可选扩展：前两项完成且时间允许时再考虑聚类、异常检测。

## 目录

```text
report/                  报告编辑稿与最终 Report.pdf
poster/                  海报编辑稿与最终 Poster.pdf
code/main.py             命令入口
code/preprocessing.py    数据检查、划分与预处理
code/classification.py   分类模型与类别不平衡处理
code/evaluation.py       评价指标、审核预算与绘图
code/pattern_mining.py   第二优先级的模式挖掘
data/                    本地原始数据及下载说明
results/figures/         正式实验图表
results/tables/          正式实验汇总表
```

`report/`、`poster/` 和结果目录目前使用 `.gitkeep` 保留空目录；尚未创建报告、海报或实验结果。

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

当前依赖清单未固定版本。首位成员验证环境后，请记录并统一实际使用的版本。

## 数据准备

按 [data/README.md](data/README.md) 下载原始数据，并放到 `data/creditcard.csv`。
原始数据不上传 GitHub，每位成员按相同说明获取。

## 当前可运行的命令

在仓库根目录执行：

```bash
python code/main.py --help
python code/main.py --check-data
```

`--check-data` 检查列名、数值、标签、缺失值、重复记录和时间范围，不修改数据。
也可以指定其他 CSV 路径：

```bash
python code/main.py --check-data --data /path/to/creditcard.csv
```

**当前是初始框架。** 数据划分、训练、评价和规则挖掘函数已预留接口，但尚未实现；数据检查通过不代表实验已完成。

## 实施顺序

1. 确认数据版本，运行数据检查。
2. 完成按 Time 顺序约 60% / 20% / 20% 的训练、验证、测试划分，并记录各部分欺诈数量；相同时间戳边界记录保留在同一部分。
3. 先用逻辑回归跑通训练和验证流程，再加入决策树、Gaussian Naive Bayes、随机森林。
4. 对代表模型比较类别权重和一种训练集重采样策略。
5. 统一计算 AP、PR 曲线，以及审核比例 0.1%、0.5%、1%、2% 下的审核数量、欺诈检出数、误报数、Precision@K、Recall@K。
6. 完成第二优先级的模式挖掘；分箱和规则发现限于训练阶段。
7. 设置与规则选择在验证阶段确定，冻结后进行最终测试，保存正式图表和汇总表。
8. 撰写报告和海报，检查新环境能否按 README 复现。

所有模型共用数据划分与评价函数。标准化、特征选择、分箱和重采样不得使用测试信息；验证集和测试集保持原有类别分布。

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

