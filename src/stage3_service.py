"""Inference contract for the stage 3 API and dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from stage2_pipeline import (
    FEATURE_COLUMNS,
    HORIZONS,
    PROJECT_ROOT,
    build_day_ahead_dataset,
    fit_horizon_models,
    load_hourly_data,
)


DEFAULT_BUNDLE = PROJECT_ROOT / "outputs" / "stage3" / "forecast_bundle.joblib"
DEFAULT_DEMO_REQUEST = PROJECT_ROOT / "outputs" / "stage3" / "demo_forecast_request.json"


def make_forecast_features(
    origin: datetime,
    history_loads: list[float],
    future_temperatures: list[float],
) -> pd.DataFrame:
    """Use exactly the historical slices used by stage 2 training."""
    if origin.hour != 0 or origin.minute != 0 or origin.second != 0 or origin.microsecond != 0:
        raise ValueError("forecast_origin 必须为整点午夜。")
    if len(history_loads) != 168 or len(future_temperatures) != 24:
        raise ValueError("需要截至预测起点的168小时负荷及未来24小时天气预报。")
    history = np.asarray(history_loads, dtype=float)
    temperatures = np.asarray(future_temperatures, dtype=float)
    if not np.isfinite(history).all() or not np.isfinite(temperatures).all():
        raise ValueError("输入不能包含 NaN 或无穷大。")

    rows = []
    for horizon, temperature in zip(HORIZONS, temperatures, strict=True):
        target = origin + timedelta(hours=horizon)
        rows.append(
            {
                "temperature_forecast_c": float(temperature),
                "target_hour_sin": np.sin(2 * np.pi * target.hour / 24),
                "target_hour_cos": np.cos(2 * np.pi * target.hour / 24),
                "target_weekday_sin": np.sin(2 * np.pi * target.weekday() / 7),
                "target_weekday_cos": np.cos(2 * np.pi * target.weekday() / 7),
                "target_is_weekend": int(target.weekday() >= 5),
                "horizon": horizon,
                "origin_load_kw": history[-1],
                "target_lag_24h": history[horizon + 143],
                "target_lag_168h": history[horizon - 1],
                "history_mean_24h": history[-24:].mean(),
                "history_std_24h": history[-24:].std(ddof=1),
                "history_mean_168h": history.mean(),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def build_demo_request(data: pd.DataFrame, origin: datetime) -> dict[str, Any]:
    matches = data.index[data["timestamp"].eq(pd.Timestamp(origin))].tolist()
    if len(matches) != 1:
        raise ValueError("演示预测起点不在数据中。")
    index = matches[0]
    if index < 167 or index + 24 >= len(data):
        raise ValueError("演示预测起点需要前168小时和后24小时数据。")
    history = data.iloc[index - 167 : index + 1]
    future = data.iloc[index + 1 : index + 25]
    return {
        "forecast_origin": origin.isoformat(),
        "history": [
            {"timestamp": row.timestamp.isoformat(), "load_kw": float(row.load_kw)}
            for row in history.itertuples(index=False)
        ],
        "future_weather": [
            {"timestamp": row.timestamp.isoformat(), "temperature_forecast_c": float(row.temperature_c)}
            for row in future.itertuples(index=False)
        ],
    }


def prepare_demo_artifacts(
    data_path: Path,
    calibration_profiles_path: Path,
    output_dir: Path,
    test_days: int = 28,
) -> dict[str, Any]:
    """Save a model trained before the held-out demo day and a replay request."""
    data = load_hourly_data(data_path)
    dataset = build_day_ahead_dataset(data)
    origins = pd.DatetimeIndex(sorted(dataset["forecast_origin"].unique()))
    if test_days < 1 or len(origins) <= test_days + 30:
        raise ValueError("测试期至少为1天，且测试期前需要至少30个训练起点。")
    cutoff = origins[-test_days]
    training = dataset.loc[dataset["target_time"] <= cutoff]
    models = fit_horizon_models(training)
    profiles = pd.read_csv(calibration_profiles_path).sort_values("horizon")
    if profiles["horizon"].tolist() != list(HORIZONS):
        raise ValueError("残差校准文件必须包含1～24小时的所有提前量。")
    demo_origin = origins[-1].to_pydatetime()
    request = build_demo_request(data, demo_origin)

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "feature_columns": FEATURE_COLUMNS,
        "residual_profiles": profiles.to_dict(orient="records"),
        "trained_through": cutoff.isoformat(),
        "demo_origin": demo_origin.isoformat(),
        "weather_assumption": "demo replay uses observed future temperature as a perfect forecast",
    }
    joblib.dump(bundle, output_dir / "forecast_bundle.joblib")
    with (output_dir / "demo_forecast_request.json").open("w", encoding="utf-8") as file:
        json.dump(request, file, ensure_ascii=False, indent=2)
    return {
        "trained_through": bundle["trained_through"],
        "demo_origin": bundle["demo_origin"],
        "history_points": len(request["history"]),
        "weather_points": len(request["future_weather"]),
    }


def load_bundle(path: Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}。请先运行 python src/prepare_stage3.py")
    bundle = joblib.load(path)
    if bundle["feature_columns"] != FEATURE_COLUMNS or set(bundle["models"]) != set(HORIZONS):
        raise ValueError("模型文件与当前特征定义不兼容，请重新运行第三阶段准备脚本。")
    return bundle


def forecast(
    bundle: dict[str, Any],
    origin: datetime,
    history_loads: list[float],
    future_temperatures: list[float],
) -> list[dict[str, Any]]:
    if origin <= datetime.fromisoformat(bundle["trained_through"]):
        raise ValueError("预测起点必须晚于模型训练截止时间。")
    features = make_forecast_features(origin, history_loads, future_temperatures)
    output = []
    for index, horizon in enumerate(HORIZONS):
        model = bundle["models"][horizon]
        prediction = float(model.predict(features.iloc[[index]])[0])
        output.append(
            {
                "target_time": (origin + timedelta(hours=horizon)).isoformat(),
                "horizon": horizon,
                "predicted_load_kw": round(prediction, 2),
                "previous_day_baseline_kw": round(float(features.iloc[index]["target_lag_24h"]), 2),
            }
        )
    return output


def score_observations(
    bundle: dict[str, Any],
    predictions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_by_horizon = {int(row["horizon"]): row for row in bundle["residual_profiles"]}
    prediction_by_time = {row["target_time"]: row for row in predictions}
    results = []
    for observed in observations:
        target_time = observed["timestamp"]
        predicted = prediction_by_time[target_time]
        profile = profile_by_horizon[predicted["horizon"]]
        residual = float(observed["load_kw"]) - predicted["predicted_load_kw"]
        score = abs(residual - profile["residual_median_kw"]) / profile["robust_scale_kw"]
        level = "critical" if score >= 5 else "high" if score >= 3.5 else "medium" if score >= 2.5 else "normal"
        results.append(
            {
                "target_time": target_time,
                "horizon": predicted["horizon"],
                "observed_load_kw": round(float(observed["load_kw"]), 2),
                "predicted_load_kw": predicted["predicted_load_kw"],
                "residual_kw": round(residual, 2),
                "score": round(float(score), 3),
                "level": level,
                "is_alert": bool(score >= 3.5),
                "alert_type": "unexpected_increase" if residual >= 0 else "unexpected_drop",
            }
        )
    return results
