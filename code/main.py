"""Project entry point. Only the initial data audit is implemented."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Shared proposed settings. Training will use these when implemented.
SEED = 42
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
REVIEW_FRACTIONS = (0.001, 0.005, 0.01, 0.02)


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
    args = parser.parse_args()
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

