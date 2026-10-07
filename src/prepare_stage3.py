"""Create held-out demo model and request for the API/dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2_pipeline import PROJECT_ROOT
from stage3_service import prepare_demo_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="准备第三阶段演示模型与请求")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "demo_load.csv")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "stage2" / "residual_profiles.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "stage3")
    parser.add_argument("--test-days", type=int, default=28)
    args = parser.parse_args()
    result = prepare_demo_artifacts(
        args.data.resolve(), args.profiles.resolve(), args.output_dir.resolve(), args.test_days
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
