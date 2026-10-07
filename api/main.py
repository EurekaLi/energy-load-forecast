"""HTTP interface for day-ahead load forecasts and residual alerts."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stage3_service import DEFAULT_BUNDLE, forecast, load_bundle, score_observations


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HistoryPoint(StrictModel):
    timestamp: datetime
    load_kw: float = Field(ge=0)


class WeatherPoint(StrictModel):
    timestamp: datetime
    temperature_forecast_c: float


class ForecastRequest(StrictModel):
    forecast_origin: datetime
    history: list[HistoryPoint] = Field(min_length=168, max_length=168)
    future_weather: list[WeatherPoint] = Field(min_length=24, max_length=24)

    @model_validator(mode="after")
    def validate_timeline(self) -> "ForecastRequest":
        origin = self.forecast_origin
        if origin.tzinfo is not None:
            raise ValueError("时间戳必须是不带时区的本地时间；请统一数据时区后提交。")
        if (origin.hour, origin.minute, origin.second, origin.microsecond) != (0, 0, 0, 0):
            raise ValueError("forecast_origin 必须是整点午夜。")
        expected_history_start = origin - timedelta(hours=167)
        for index, item in enumerate(self.history):
            if item.timestamp != expected_history_start + timedelta(hours=index):
                raise ValueError("history 必须按小时连续排列，并以 forecast_origin 结束。")
        for index, item in enumerate(self.future_weather, start=1):
            if item.timestamp != origin + timedelta(hours=index):
                raise ValueError("future_weather 必须覆盖预测起点之后连续24小时。")
        return self


class Observation(StrictModel):
    timestamp: datetime
    load_kw: float = Field(ge=0)


class AnomalyRequest(StrictModel):
    forecast: ForecastRequest
    observations: list[Observation] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def validate_observations(self) -> "AnomalyRequest":
        origin = self.forecast.forecast_origin
        valid_times = {origin + timedelta(hours=h) for h in range(1, 25)}
        times = [row.timestamp for row in self.observations]
        if len(set(times)) != len(times) or not set(times).issubset(valid_times):
            raise ValueError("observations 时间戳必须唯一，且位于未来24小时内。")
        return self


@lru_cache(maxsize=1)
def get_model_bundle() -> dict:
    bundle_path = Path(os.environ.get("ENERGY_MODEL_BUNDLE", str(DEFAULT_BUNDLE)))
    return load_bundle(bundle_path)


app = FastAPI(title="能源负荷预测与异常检测", version="0.3.0")


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "not_ready", "detail": str(exc)}
    return {"status": "ok", "trained_through": bundle["trained_through"]}


@app.get("/model-info")
def model_info() -> dict:
    bundle = _ready_bundle()
    return {
        "trained_through": bundle["trained_through"],
        "horizons": list(range(1, 25)),
        "features": bundle["feature_columns"],
        "weather_assumption": bundle["weather_assumption"],
        "input_contract": "168 hourly historical loads ending at midnight and 24 hourly weather forecasts",
    }

def _ready_bundle() -> dict:
    try:
        return get_model_bundle()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _forecast(request: ForecastRequest) -> tuple[dict, list[dict]]:
    bundle = _ready_bundle()
    try:
        predictions = forecast(
            bundle,
            request.forecast_origin,
            [item.load_kw for item in request.history],
            [item.temperature_forecast_c for item in request.future_weather],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return bundle, predictions


@app.post("/forecast")
def forecast_endpoint(request: ForecastRequest) -> dict:
    bundle, predictions = _forecast(request)
    return {
        "forecast_origin": request.forecast_origin.isoformat(),
        "model_trained_through": bundle["trained_through"],
        "predictions": predictions,
    }


@app.post("/anomalies")
def anomalies_endpoint(request: AnomalyRequest) -> dict:
    bundle, predictions = _forecast(request.forecast)
    observations = [
        {"timestamp": row.timestamp.isoformat(), "load_kw": row.load_kw}
        for row in request.observations
    ]
    scored = score_observations(bundle, predictions, observations)
    return {
        "forecast_origin": request.forecast.forecast_origin.isoformat(),
        "alert_count": sum(row["is_alert"] for row in scored),
        "results": scored,
    }
