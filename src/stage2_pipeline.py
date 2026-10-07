"""Leakage-safe day-ahead forecasting, rolling backtest and anomaly detection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2"
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


HORIZONS = tuple(range(1, 25))
FEATURE_COLUMNS = [
    "temperature_forecast_c",
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_is_weekend",
    "horizon",
    "origin_load_kw",
    "target_lag_24h",
    "target_lag_168h",
    "history_mean_24h",
    "history_std_24h",
    "history_mean_168h",
]


def load_hourly_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "load_kw", "temperature_c"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{sorted(missing)}")
    data = data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if data[list(required)].isna().any().any():
        raise ValueError("必要列存在缺失值，请先清洗或插值。")
    if not data["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("timestamp 必须是连续的小时级时间序列。")
    return data


def build_day_ahead_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Convert hourly data into 24 direct-forecast samples per midnight origin.

    The forecast is issued immediately after the load at ``forecast_origin`` is
    observed. Every load-derived feature therefore points to that time or older.
    Future temperature is treated as a perfect weather forecast in demo data.
    """
    records: list[dict[str, Any]] = []
    loads = data["load_kw"].to_numpy(dtype=float)
    temperatures = data["temperature_c"].to_numpy(dtype=float)
    timestamps = data["timestamp"].reset_index(drop=True)

    for origin_index in range(167, len(data) - 24):
        origin_time = timestamps.iloc[origin_index]
        if origin_time.hour != 0:
            continue

        history_24 = loads[origin_index - 23 : origin_index + 1]
        history_168 = loads[origin_index - 167 : origin_index + 1]
        for horizon in HORIZONS:
            target_index = origin_index + horizon
            target_time = timestamps.iloc[target_index]
            target_hour = target_time.hour
            target_weekday = target_time.dayofweek
            records.append(
                {
                    "forecast_origin": origin_time,
                    "target_time": target_time,
                    "horizon": horizon,
                    "actual_load_kw": loads[target_index],
                    "temperature_forecast_c": temperatures[target_index],
                    "target_hour_sin": np.sin(2 * np.pi * target_hour / 24),
                    "target_hour_cos": np.cos(2 * np.pi * target_hour / 24),
                    "target_weekday_sin": np.sin(2 * np.pi * target_weekday / 7),
                    "target_weekday_cos": np.cos(2 * np.pi * target_weekday / 7),
                    "target_is_weekend": int(target_weekday >= 5),
                    "origin_load_kw": loads[origin_index],
                    "target_lag_24h": loads[target_index - 24],
                    "target_lag_168h": loads[target_index - 168],
                    "history_mean_24h": float(history_24.mean()),
                    "history_std_24h": float(history_24.std(ddof=1)),
                    "history_mean_168h": float(history_168.mean()),
                }
            )

    dataset = pd.DataFrame.from_records(records)
    if dataset.empty:
        raise ValueError("数据不足：至少需要 9 天小时数据，并包含午夜预测起点。")
    return dataset


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.055,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=8,
        l2_regularization=1.0,
        random_state=42,
    )


def fit_horizon_models(train: pd.DataFrame) -> dict[int, HistGradientBoostingRegressor]:
    models: dict[int, HistGradientBoostingRegressor] = {}
    for horizon in HORIZONS:
        subset = train.loc[train["horizon"] == horizon]
        if len(subset) < 30:
            raise ValueError(f"horizon={horizon} 只有 {len(subset)} 条训练样本，至少需要 30 条。")
        model = _new_model()
        model.fit(subset[FEATURE_COLUMNS], subset["actual_load_kw"])
        models[horizon] = model
    return models


def predict_horizons(
    models: dict[int, HistGradientBoostingRegressor], rows: pd.DataFrame
) -> pd.DataFrame:
    result = rows.copy()
    result["predicted_load_kw"] = np.nan
    for horizon, model in models.items():
        mask = result["horizon"] == horizon
        result.loc[mask, "predicted_load_kw"] = model.predict(result.loc[mask, FEATURE_COLUMNS])
    result["baseline_load_kw"] = result["target_lag_24h"]
    result["residual_kw"] = result["actual_load_kw"] - result["predicted_load_kw"]
    return result


def rolling_backtest(
    dataset: pd.DataFrame, origins: pd.DatetimeIndex, fold_days: int = 7
) -> pd.DataFrame:
    """Expanding-window backtest; refit at the beginning of every fold."""
    predictions: list[pd.DataFrame] = []
    for start in range(0, len(origins), fold_days):
        fold_origins = origins[start : start + fold_days]
        cutoff = fold_origins[0]
        # At cutoff, only labels whose target time is no later than cutoff exist.
        train = dataset.loc[dataset["target_time"] <= cutoff]
        test = dataset.loc[dataset["forecast_origin"].isin(fold_origins)]
        models = fit_horizon_models(train)
        fold_result = predict_horizons(models, test)
        fold_result["model_train_cutoff"] = cutoff
        predictions.append(fold_result)
    return pd.concat(predictions, ignore_index=True).sort_values("target_time").reset_index(drop=True)


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_array = actual.to_numpy(dtype=float)
    predicted_array = predicted.to_numpy(dtype=float)
    nonzero = np.abs(actual_array) > 1e-8
    mape = np.mean(np.abs((actual_array[nonzero] - predicted_array[nonzero]) / actual_array[nonzero])) * 100
    return {
        "mae_kw": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse_kw": float(mean_squared_error(actual_array, predicted_array) ** 0.5),
        "mape_percent": float(mape),
    }


def horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for horizon, group in predictions.groupby("horizon", sort=True):
        model = regression_metrics(group["actual_load_kw"], group["predicted_load_kw"])
        baseline = regression_metrics(group["actual_load_kw"], group["baseline_load_kw"])
        rows.append(
            {
                "horizon": int(horizon),
                "model_mae_kw": model["mae_kw"],
                "model_rmse_kw": model["rmse_kw"],
                "model_mape_percent": model["mape_percent"],
                "baseline_mae_kw": baseline["mae_kw"],
                "baseline_rmse_kw": baseline["rmse_kw"],
                "baseline_mape_percent": baseline["mape_percent"],
                "model_beats_baseline": model["mae_kw"] < baseline["mae_kw"],
            }
        )
    return pd.DataFrame(rows)


def segment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["segment"] = np.select(
        [frame["target_is_weekend"].eq(1), frame["target_time"].dt.hour.between(8, 20)],
        ["weekend", "weekday_peak"],
        default="weekday_offpeak",
    )
    rows = []
    for segment, group in frame.groupby("segment"):
        values = regression_metrics(group["actual_load_kw"], group["predicted_load_kw"])
        rows.append({"segment": segment, "samples": len(group), **values})
    return pd.DataFrame(rows)


def fit_residual_profiles(calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in calibration.groupby("horizon", sort=True):
        residual = group["residual_kw"].to_numpy(dtype=float)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        rows.append(
            {
                "horizon": int(horizon),
                "residual_median_kw": median,
                "residual_mad_kw": mad,
                "robust_scale_kw": max(1.4826 * mad, 5.0),
                "calibration_samples": len(group),
            }
        )
    return pd.DataFrame(rows)


def inject_demo_anomalies(predictions: pd.DataFrame) -> pd.DataFrame:
    """Inject labeled spikes/drops after forecasting, solely for detector evaluation."""
    result = predictions.sort_values("target_time").reset_index(drop=True).copy()
    result["observed_load_kw"] = result["actual_load_kw"]
    result["injected_anomaly_type"] = "normal"
    if len(result) < 100:
        raise ValueError("异常注入评估至少需要 100 个测试点。")

    injections = [
        (int(len(result) * 0.18), 1, 1.38, "load_spike"),
        (int(len(result) * 0.43), 1, 0.62, "load_drop"),
        (int(len(result) * 0.68), 3, 1.32, "consecutive_high"),
    ]
    for start, duration, factor, label in injections:
        positions = list(range(start, min(start + duration, len(result))))
        result.loc[positions, "observed_load_kw"] *= factor
        result.loc[positions, "injected_anomaly_type"] = label
    result["is_injected_anomaly"] = result["injected_anomaly_type"].ne("normal")
    return result


def score_anomalies(observations: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    result = observations.merge(profiles, on="horizon", how="left", validate="many_to_one")
    result["observed_residual_kw"] = result["observed_load_kw"] - result["predicted_load_kw"]
    result["anomaly_score"] = (
        (result["observed_residual_kw"] - result["residual_median_kw"]).abs()
        / result["robust_scale_kw"]
    )
    result["anomaly_level"] = np.select(
        [result["anomaly_score"] >= 5.0, result["anomaly_score"] >= 3.5, result["anomaly_score"] >= 2.5],
        ["critical", "high", "medium"],
        default="normal",
    )
    result["is_detected_anomaly"] = result["anomaly_score"] >= 3.5
    result["alert_type"] = np.where(
        result["observed_residual_kw"] >= 0, "unexpected_increase", "unexpected_drop"
    )
    result["likely_reason"] = np.where(
        result["observed_residual_kw"] >= 0,
        "actual load is far above the forecast",
        "actual load is far below the forecast",
    )
    return result


def anomaly_evaluation(scored: pd.DataFrame) -> dict[str, float | int]:
    expected = scored["is_injected_anomaly"].astype(bool)
    detected = scored["is_detected_anomaly"].astype(bool)
    tp = int((expected & detected).sum())
    fp = int((~expected & detected).sum())
    fn = int((expected & ~detected).sum())
    tn = int((~expected & ~detected).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_anomaly_events(scored: pd.DataFrame) -> pd.DataFrame:
    detected = scored.loc[scored["is_detected_anomaly"]].sort_values("target_time").copy()
    columns = [
        "event_id", "start_time", "end_time", "duration_hours", "alert_type",
        "max_score", "level", "max_abs_residual_kw", "contains_injected_anomaly", "likely_reason",
    ]
    if detected.empty:
        return pd.DataFrame(columns=columns)

    new_event = detected["target_time"].diff().gt(pd.Timedelta(hours=1))
    detected["event_id"] = new_event.cumsum() + 1
    severity = {"normal": 0, "medium": 1, "high": 2, "critical": 3}
    inverse_severity = {value: key for key, value in severity.items()}
    events = []
    for event_id, group in detected.groupby("event_id"):
        dominant_type = group["alert_type"].mode().iat[0]
        events.append(
            {
                "event_id": int(event_id),
                "start_time": group["target_time"].min(),
                "end_time": group["target_time"].max(),
                "duration_hours": len(group),
                "alert_type": dominant_type,
                "max_score": float(group["anomaly_score"].max()),
                "level": inverse_severity[max(group["anomaly_level"].map(severity))],
                "max_abs_residual_kw": float(group["observed_residual_kw"].abs().max()),
                "contains_injected_anomaly": bool(group["is_injected_anomaly"].any()),
                "likely_reason": group.loc[group["alert_type"].eq(dominant_type), "likely_reason"].iat[0],
            }
        )
    return pd.DataFrame(events, columns=columns)


def _json_dump(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_plots(
    test_predictions: pd.DataFrame,
    per_horizon: pd.DataFrame,
    calibration: pd.DataFrame,
    scored: pd.DataFrame,
    output_dir: Path,
) -> None:
    sample = test_predictions.tail(7 * 24)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sample["target_time"], sample["actual_load_kw"], label="Actual", linewidth=1.8)
    ax.plot(sample["target_time"], sample["predicted_load_kw"], label="24-model direct forecast", linewidth=1.3)
    ax.plot(sample["target_time"], sample["baseline_load_kw"], label="Previous-day baseline", alpha=0.65)
    ax.set(title="Day-ahead rolling backtest - last 7 days", ylabel="Load (kW)", xlabel="Target time")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "backtest_forecast.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(per_horizon["horizon"], per_horizon["model_mae_kw"], marker="o", label="Model")
    ax.plot(per_horizon["horizon"], per_horizon["baseline_mae_kw"], marker="o", label="Baseline")
    ax.set(title="MAE by forecast horizon", xlabel="Forecast horizon (hours)", ylabel="MAE (kW)", xticks=range(1, 25))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "horizon_mae.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(calibration["residual_kw"], bins=30, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(title="Out-of-sample calibration residuals", xlabel="Residual (kW)", ylabel="Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(scored["target_time"], scored["observed_load_kw"], label="Observed", linewidth=1.5)
    ax.plot(scored["target_time"], scored["predicted_load_kw"], label="Forecast", linewidth=1.1)
    anomalies = scored.loc[scored["is_detected_anomaly"]]
    ax.scatter(anomalies["target_time"], anomalies["observed_load_kw"], color="red", marker="x", s=60, label="Detected anomaly")
    ax.set(title="Residual anomaly detection", xlabel="Time", ylabel="Load (kW)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_timeline.png", dpi=150)
    plt.close(fig)


def run_stage2(
    data_path: Path,
    output_dir: Path,
    calibration_days: int = 28,
    test_days: int = 28,
    fold_days: int = 7,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_day_ahead_dataset(load_hourly_data(data_path))
    origins = pd.DatetimeIndex(sorted(dataset["forecast_origin"].unique()))
    required_origins = calibration_days + test_days + 35
    if len(origins) < required_origins:
        raise ValueError(f"有效预测起点只有 {len(origins)} 天，至少需要 {required_origins} 天。")

    calibration_origins = origins[-(calibration_days + test_days) : -test_days]
    test_origins = origins[-test_days:]
    calibration = rolling_backtest(dataset, calibration_origins, fold_days)
    test_predictions = rolling_backtest(dataset, test_origins, fold_days)

    overall_model = regression_metrics(test_predictions["actual_load_kw"], test_predictions["predicted_load_kw"])
    overall_baseline = regression_metrics(test_predictions["actual_load_kw"], test_predictions["baseline_load_kw"])
    per_horizon = horizon_metrics(test_predictions)
    per_segment = segment_metrics(test_predictions)
    profiles = fit_residual_profiles(calibration)
    scored = score_anomalies(inject_demo_anomalies(test_predictions), profiles)
    events = build_anomaly_events(scored)
    anomaly_metrics = anomaly_evaluation(scored)

    summary: dict[str, Any] = {
        "strategy": "24 independent direct models",
        "weather_assumption": "observed future temperature is used as a perfect forecast in demo data",
        "calibration_period": [str(calibration["target_time"].min()), str(calibration["target_time"].max())],
        "test_period": [str(test_predictions["target_time"].min()), str(test_predictions["target_time"].max())],
        "rolling_fold_days": fold_days,
        "model": overall_model,
        "previous_day_baseline": overall_baseline,
        "horizons_beating_baseline": int(per_horizon["model_beats_baseline"].sum()),
        "total_horizons": 24,
        "anomaly_detection": anomaly_metrics,
    }

    output_columns = [
        "forecast_origin", "target_time", "horizon", "actual_load_kw", "predicted_load_kw",
        "baseline_load_kw", "residual_kw", "model_train_cutoff",
    ]
    calibration[output_columns].to_csv(output_dir / "calibration_predictions.csv", index=False)
    test_predictions[output_columns].to_csv(output_dir / "backtest_predictions.csv", index=False)
    per_horizon.to_csv(output_dir / "horizon_metrics.csv", index=False)
    per_segment.to_csv(output_dir / "segment_metrics.csv", index=False)
    profiles.to_csv(output_dir / "residual_profiles.csv", index=False)
    scored.to_csv(output_dir / "anomaly_scores.csv", index=False)
    events.to_csv(output_dir / "anomaly_events.csv", index=False)
    _json_dump(summary, output_dir / "backtest_summary.json")
    _json_dump(anomaly_metrics, output_dir / "anomaly_evaluation.json")

    final_models = fit_horizon_models(dataset)
    joblib.dump(
        {
            "models": final_models,
            "feature_columns": FEATURE_COLUMNS,
            "horizons": HORIZONS,
            "forecast_definition": "forecast at midnight after origin load, targets +1h through +24h",
        },
        output_dir / "day_ahead_24_models.joblib",
    )
    save_plots(test_predictions, per_horizon, calibration, scored, output_dir)
    return summary
