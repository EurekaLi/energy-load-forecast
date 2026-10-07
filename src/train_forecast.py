"""Train and evaluate a day-ahead-style hourly load forecasting baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "demo_load.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Keep matplotlib cache inside the project so the script also works in sandboxes.
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET = "load_kw"
FEATURE_COLUMNS = [
    "temperature_c",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_168h",
]


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", TARGET, "temperature_c"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{sorted(missing)}")

    data = data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if data[list(required)].isna().any().any():
        raise ValueError("必要列存在缺失值，请先清洗或插值。")
    if not data["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("timestamp 必须是连续的小时级时间序列。")
    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe calendar, lag and historical rolling features."""
    frame = data.copy()
    hour = frame["timestamp"].dt.hour
    weekday = frame["timestamp"].dt.dayofweek

    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    frame["is_weekend"] = (weekday >= 5).astype(int)
    frame["lag_1h"] = frame[TARGET].shift(1)
    frame["lag_24h"] = frame[TARGET].shift(24)
    frame["lag_168h"] = frame[TARGET].shift(168)

    history = frame[TARGET].shift(1)
    frame["rolling_mean_24h"] = history.rolling(24).mean()
    frame["rolling_std_24h"] = history.rolling(24).std()
    frame["rolling_mean_168h"] = history.rolling(168).mean()
    return frame.dropna().reset_index(drop=True)


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    actual_array = actual.to_numpy()
    nonzero = np.abs(actual_array) > 1e-8
    mape = np.mean(np.abs((actual_array[nonzero] - predicted[nonzero]) / actual_array[nonzero])) * 100
    return {
        "mae_kw": float(mean_absolute_error(actual_array, predicted)),
        "rmse_kw": float(mean_squared_error(actual_array, predicted) ** 0.5),
        "mape_percent": float(mape),
    }


def save_plot(result: pd.DataFrame, path: Path) -> None:
    sample = result.tail(7 * 24)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sample["timestamp"], sample["actual_load_kw"], label="Actual", linewidth=1.8)
    ax.plot(sample["timestamp"], sample["model_prediction_kw"], label="Model", linewidth=1.4)
    ax.plot(sample["timestamp"], sample["yesterday_baseline_kw"], label="Previous-day baseline", alpha=0.65)
    ax.set(title="Hourly load forecast - last 7 test days", ylabel="Load (kW)", xlabel="Time")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练小时级负荷预测模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="输入 CSV")
    parser.add_argument("--test-days", type=int, default=14, help="测试集天数")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = build_features(load_data(data_path))
    test_rows = args.test_days * 24
    if args.test_days < 1 or len(frame) <= test_rows:
        raise ValueError("测试集必须至少 1 天，且训练集不能为空。")

    train = frame.iloc[:-test_rows]
    test = frame.iloc[-test_rows:]
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.5,
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET])

    model_prediction = model.predict(test[FEATURE_COLUMNS])
    baseline_prediction = test["lag_24h"].to_numpy()
    metrics = {
        "train_period": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "test_period": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        "train_rows": len(train),
        "test_rows": len(test),
        "model": regression_metrics(test[TARGET], model_prediction),
        "previous_day_baseline": regression_metrics(test[TARGET], baseline_prediction),
    }

    predictions = pd.DataFrame(
        {
            "timestamp": test["timestamp"],
            "actual_load_kw": test[TARGET],
            "model_prediction_kw": model_prediction.round(2),
            "yesterday_baseline_kw": baseline_prediction.round(2),
        }
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS, "target": TARGET},
        output_dir / "load_forecast_model.joblib",
    )
    save_plot(predictions, output_dir / "forecast_comparison.png")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"结果已保存到：{output_dir}")


if __name__ == "__main__":
    main()
