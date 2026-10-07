"""Stage 3 integration tests for feature parity and the HTTP contract."""

from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app
from stage2_pipeline import FEATURE_COLUMNS, build_day_ahead_dataset, load_hourly_data
from stage3_service import DEFAULT_DEMO_REQUEST, make_forecast_features


class Stage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.request = json.loads(DEFAULT_DEMO_REQUEST.read_text(encoding="utf-8"))
        cls.client = TestClient(app)

    def test_inference_features_equal_training_features(self) -> None:
        data = load_hourly_data(PROJECT_ROOT / "data" / "demo_load.csv")
        dataset = build_day_ahead_dataset(data)
        expected = dataset.loc[
            dataset["forecast_origin"].eq(self.request["forecast_origin"]), FEATURE_COLUMNS
        ].sort_values("horizon").to_numpy(dtype=float)
        from datetime import datetime

        actual = make_forecast_features(
            datetime.fromisoformat(self.request["forecast_origin"]),
            [row["load_kw"] for row in self.request["history"]],
            [row["temperature_forecast_c"] for row in self.request["future_weather"]],
        ).to_numpy(dtype=float)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_forecast_returns_24_targets(self) -> None:
        response = self.client.post("/forecast", json=self.request)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(len(payload["predictions"]), 24)
        self.assertEqual([row["horizon"] for row in payload["predictions"]], list(range(1, 25)))

    def test_rejects_non_contiguous_history(self) -> None:
        broken = deepcopy(self.request)
        broken["history"][9]["timestamp"] = broken["history"][8]["timestamp"]
        response = self.client.post("/forecast", json=broken)
        self.assertEqual(response.status_code, 422)

    def test_rejects_prediction_before_model_cutoff(self) -> None:
        broken = deepcopy(self.request)
        broken["forecast_origin"] = "2025-06-01T00:00:00"
        from datetime import datetime, timedelta

        origin = datetime.fromisoformat(broken["forecast_origin"])
        for i, row in enumerate(broken["history"]):
            row["timestamp"] = (origin - timedelta(hours=167 - i)).isoformat()
        for i, row in enumerate(broken["future_weather"], start=1):
            row["timestamp"] = (origin + timedelta(hours=i)).isoformat()
        response = self.client.post("/forecast", json=broken)
        self.assertEqual(response.status_code, 422)

    def test_anomaly_endpoint_detects_large_spike(self) -> None:
        forecast = self.client.post("/forecast", json=self.request).json()
        first = forecast["predictions"][0]
        response = self.client.post(
            "/anomalies",
            json={
                "forecast": self.request,
                "observations": [
                    {"timestamp": first["target_time"], "load_kw": first["predicted_load_kw"] * 1.5}
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["alert_count"], 1)
        self.assertEqual(payload["results"][0]["alert_type"], "unexpected_increase")


if __name__ == "__main__":
    unittest.main()
