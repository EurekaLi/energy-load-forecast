"""Check unit conversion and forecast-origin alignment in the real-data experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "real_load_benchmark"
sys.path.insert(0, str(EXPERIMENT_DIR))

from benchmark import build_samples, decompose, fit_dlinear, predict_dlinear
from prepare_data import to_hourly


class RealLoadBenchmarkTests(unittest.TestCase):
    def test_hourly_power_is_mean_not_sum(self) -> None:
        quarter_hours = pd.date_range("2014-04-01", periods=8, freq="15min")
        raw = pd.DataFrame({"timestamp": quarter_hours, "MT_001": [2, 4, 6, 8, 10, 12, 14, 16]})
        hourly = to_hourly(raw)
        np.testing.assert_allclose(hourly["MT_001"], [5, 13])

    def test_lags_and_targets_are_aligned_at_midnight(self) -> None:
        index = pd.date_range("2014-04-01", periods=24 * 20, freq="h")
        series = pd.Series(np.arange(len(index), dtype=float) + 1, index=index)
        tabular, sequences = build_samples(series)
        first_origin = tabular["origin"].min()
        first = tabular.loc[tabular["origin"].eq(first_origin)].sort_values("horizon")
        self.assertEqual(len(first), 24)
        self.assertEqual(first_origin.hour, 0)
        for row in first.itertuples():
            self.assertEqual(row.target_time, first_origin + pd.Timedelta(hours=row.horizon))
            self.assertEqual(row.target_lag_24h, series[row.target_time - pd.Timedelta(hours=24)])
            self.assertEqual(row.target_lag_168h, series[row.target_time - pd.Timedelta(hours=168)])
            self.assertLessEqual(row.target_time - pd.Timedelta(hours=24), first_origin)
        self.assertEqual(len(sequences.iloc[0]["history"]), 168)
        self.assertEqual(len(sequences.iloc[0]["future"]), 24)

    def test_dlinear_decomposition_and_output_shape(self) -> None:
        history = np.linspace(1, 168, 168)
        transformed = decompose(history)
        np.testing.assert_allclose(transformed[:168] + transformed[168:], history)
        index = pd.date_range("2014-04-01", periods=24 * 60, freq="h")
        series = pd.Series(100 + 10 * np.sin(np.arange(len(index)) / 24), index=index)
        _, sequences = build_samples(series)
        train = sequences.iloc[:35]
        test = sequences.iloc[35:37]
        model = fit_dlinear(train, 100, 10)
        self.assertEqual(predict_dlinear(model, test, 100, 10).shape, (2, 24))


if __name__ == "__main__":
    unittest.main()
