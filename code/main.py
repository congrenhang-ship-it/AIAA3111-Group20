"""负责人 A：运行入口与流程整合。

任务：统一数据路径、随机种子、时间划分和审核预算；接入 B/C/D 的模块。
路线：数据 → 分类/聚类/异常检测/规则 → 验证冻结 → 测试 → 时间分析与图表。
当前可用：帮助、任务清单、数据检查。完整实验调度尚未实现。
"""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Shared proposed settings. Training will use these when implemented.
SEED = 42
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
REVIEW_FRACTIONS = (0.001, 0.005, 0.01, 0.02)
TIME_WINDOW_SECONDS = 3600  # 初始建议；窗口口径须在最终测试前确定。

TASKS = {
    "A": ["main.py: 流程整合", "preprocessing.py: 数据与时间划分", "temporal.py: 时间窗口分析"],
    "B": ["classification.py: 分类模型、不平衡处理、风险分数"],
    "C": ["unsupervised.py: K-means、Isolation Forest、簇距离", "evaluation.py: 评价、等预算组合、图表"],
    "D": ["pattern_mining.py: 分箱、Apriori、规则验证", "report/、poster/: 文档整合，全组提供内容"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Group 20 credit card fraud project: initial framework."
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data" / "creditcard.csv",
        help="Path to the original creditcard.csv file.",
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Run a read-only data audit."
    )
    parser.add_argument("--tasks", action="store_true", help="Show agreed module ownership.")
    args = parser.parse_args()
    if args.tasks:
        print(json.dumps(TASKS, indent=2, ensure_ascii=False))
        if not args.check_data:
            return 0
    if not args.check_data:
        parser.print_help()
        return 0
    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}. See data/README.md.")
    try:
        from preprocessing import audit_data, load_data

        summary = audit_data(load_data(args.data))
    except ImportError:
        parser.error("Install dependencies with: python -m pip install -r requirements.txt")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

