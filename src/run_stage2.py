"""Command-line entry point for stage 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2_pipeline import DEFAULT_OUTPUT_DIR, PROJECT_ROOT, run_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行日前24小时预测、滚动回测和残差异常检测")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo_load.csv",
        help="小时级负荷 CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--calibration-days", type=int, default=28, help="异常阈值校准天数")
    parser.add_argument("--test-days", type=int, default=28, help="最终回测天数")
    parser.add_argument("--fold-days", type=int, default=7, help="滚动重训间隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stage2(
        data_path=args.data.resolve(),
        output_dir=args.output_dir.resolve(),
        calibration_days=args.calibration_days,
        test_days=args.test_days,
        fold_days=args.fold_days,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"第二阶段结果已保存到：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
