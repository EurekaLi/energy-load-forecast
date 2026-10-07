"""Generate reproducible hourly load data for the first project milestone."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "demo_load.csv"


def generate_demo_data(days: int, seed: int) -> pd.DataFrame:
    """Build an hourly data set with daily, weekly and weather-driven patterns."""
    if days < 30:
        raise ValueError("days 至少为 30，才能进行可靠的时间切分。")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    day_index = np.arange(len(timestamps)) / 24

    seasonal_temperature = 17 + 10 * np.sin(2 * np.pi * (day_index - 35) / 365)
    daily_temperature = 4 * np.sin(2 * np.pi * (hour - 14) / 24)
    temperature = seasonal_temperature + daily_temperature + rng.normal(0, 1.3, len(timestamps))

    morning_peak = 105 * np.exp(-0.5 * ((hour - 9) / 2.2) ** 2)
    evening_peak = 155 * np.exp(-0.5 * ((hour - 19) / 2.8) ** 2)
    business_load = np.where((hour >= 8) & (hour <= 18), 75, 0)
    weekend_effect = np.where(weekday >= 5, -65, 0)
    cooling_heating_load = 4.8 * np.abs(temperature - 20)
    noise = rng.normal(0, 13, len(timestamps))

    load_kw = 330 + morning_peak + evening_peak + business_load + weekend_effect
    load_kw = load_kw + cooling_heating_load + noise
    load_kw = np.maximum(load_kw, 80).round(2)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "load_kw": load_kw,
            "temperature_c": temperature.round(2),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成小时级演示负荷数据")
    parser.add_argument("--days", type=int, default=180, help="生成天数，默认 180")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV 输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_demo_data(args.days, args.seed)
    data.to_csv(output, index=False)
    print(f"已生成 {len(data):,} 条数据：{output}")
    print(data.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
