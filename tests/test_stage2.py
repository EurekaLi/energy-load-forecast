"""Small regression tests for leakage and anomaly scoring."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_demo_data import generate_demo_data
from stage2_pipeline import (
    build_day_ahead_dataset,
    fit_residual_profiles,
    inject_demo_anomalies,
    score_anomalies,
)


class Stage2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_day_ahead_dataset(generate_demo_data(days=100, seed=7))

    def test_targets_match_horizons(self) -> None:
        hours = (
            (self.dataset["target_time"] - self.dataset["forecast_origin"])
            / pd.Timedelta(hours=1)
        )
        self.assertTrue(hours.eq(self.dataset["horizon"]).all())
        self.assertEqual(set(self.dataset["horizon"]), set(range(1, 25)))

    def test_known_lags_never_cross_forecast_origin(self) -> None:
        target_lag_24_time = self.dataset["target_time"] - pd.Timedelta(hours=24)
        target_lag_168_time = self.dataset["target_time"] - pd.Timedelta(hours=168)
        self.assertTrue(target_lag_24_time.le(self.dataset["forecast_origin"]).all())
        self.assertTrue(target_lag_168_time.le(self.dataset["forecast_origin"]).all())

    def test_injected_anomalies_receive_scores(self) -> None:
        sample = self.dataset.tail(10 * 24).copy()
        sample["predicted_load_kw"] = sample["actual_load_kw"]
        sample["residual_kw"] = 0.0
        profiles = fit_residual_profiles(sample)
        scored = score_anomalies(inject_demo_anomalies(sample), profiles)
        injected = scored.loc[scored["is_injected_anomaly"]]
        self.assertTrue(injected["is_detected_anomaly"].all())


if __name__ == "__main__":
    unittest.main()
