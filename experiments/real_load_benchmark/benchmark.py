"""Compare regression and sequence forecasting on real UCI load curves."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "uci_hourly_selected.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "real_load_benchmark"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT / ".matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FOLDS = [
    ("2014-08-01 00:00:00", "2014-08-14 00:00:00"),
    ("2014-09-01 00:00:00", "2014-09-14 00:00:00"),
    ("2014-09-15 00:00:00", "2014-09-28 00:00:00"),
]
TABULAR_FEATURES = [
    "horizon", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "is_weekend", "origin_load_kw", "target_lag_24h", "target_lag_168h",
    "mean_24h", "std_24h", "mean_168h",
]
TABULAR_LAG_FEATURES = [
    "horizon", "origin_load_kw", "target_lag_24h", "target_lag_168h",
    "mean_24h", "std_24h", "mean_168h",
]
MODEL_NAMES = [
    "yesterday", "last_week", "ridge_lags", "ridge", "hist_gb",
    "direct_linear", "dlinear", "sarima",
]


def load_hourly(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    if data.index.has_duplicates or not data.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("处理后的数据必须是连续、无重复的小时序列。")
    if data.isna().any().any():
        raise ValueError("处理后的数据存在缺失值。")
    return data


def build_samples(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make the same midnight origins for every forecasting method."""
    values = series.to_numpy(dtype=float)
    timestamps = series.index
    tabular_rows = []
    sequence_rows = []
    for origin_index in range(167, len(values) - 24):
        origin = timestamps[origin_index]
        if origin.hour != 0:
            continue
        history = values[origin_index - 167 : origin_index + 1]
        future = values[origin_index + 1 : origin_index + 25]
        sequence_rows.append({"origin": origin, "history": history.copy(), "future": future.copy()})
        for horizon in range(1, 25):
            target_index = origin_index + horizon
            target_time = timestamps[target_index]
            hour = target_time.hour
            weekday = target_time.dayofweek
            tabular_rows.append(
                {
                    "origin": origin,
                    "target_time": target_time,
                    "horizon": horizon,
                    "actual_kw": values[target_index],
                    "hour_sin": np.sin(2 * np.pi * hour / 24),
                    "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "weekday_sin": np.sin(2 * np.pi * weekday / 7),
                    "weekday_cos": np.cos(2 * np.pi * weekday / 7),
                    "is_weekend": int(weekday >= 5),
                    "origin_load_kw": values[origin_index],
                    "target_lag_24h": values[target_index - 24],
                    "target_lag_168h": values[target_index - 168],
                    "mean_24h": float(history[-24:].mean()),
                    "std_24h": float(history[-24:].std(ddof=1)),
                    "mean_168h": float(history.mean()),
                }
            )
    return pd.DataFrame(tabular_rows), pd.DataFrame(sequence_rows)


def decompose(history: np.ndarray, window: int = 25) -> np.ndarray:
    """DLinear-style moving-average trend and residual, then concatenate."""
    pad = window // 2
    extended = np.pad(history, (pad, pad), mode="edge")
    trend = np.convolve(extended, np.ones(window) / window, mode="valid")
    seasonal = history - trend
    return np.concatenate([seasonal, trend])


def fit_dlinear(train_sequences: pd.DataFrame, training_mean: float, training_std: float) -> Ridge:
    x = np.stack(
        [decompose((row.history - training_mean) / training_std) for row in train_sequences.itertuples()]
    )
    y = np.stack([(row.future - training_mean) / training_std for row in train_sequences.itertuples()])
    model = Ridge(alpha=50.0)
    model.fit(x, y)
    return model


def predict_dlinear(model: Ridge, sequences: pd.DataFrame, training_mean: float, training_std: float) -> np.ndarray:
    x = np.stack([decompose((row.history - training_mean) / training_std) for row in sequences.itertuples()])
    return model.predict(x) * training_std + training_mean


def fit_direct_linear(train_sequences: pd.DataFrame, training_mean: float, training_std: float) -> Ridge:
    x = np.stack([(row.history - training_mean) / training_std for row in train_sequences.itertuples()])
    y = np.stack([(row.future - training_mean) / training_std for row in train_sequences.itertuples()])
    model = Ridge(alpha=50.0)
    model.fit(x, y)
    return model


def predict_direct_linear(model: Ridge, sequences: pd.DataFrame, training_mean: float, training_std: float) -> np.ndarray:
    x = np.stack([(row.history - training_mean) / training_std for row in sequences.itertuples()])
    return model.predict(x) * training_std + training_mean


def predict_sarima(series: pd.Series, cutoff: pd.Timestamp, origins: list[pd.Timestamp]) -> tuple[np.ndarray, float, bool]:
    """Fit once per fold, update observed state at each origin, forecast 24 hours."""
    training = series.loc[:cutoff]
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        fit = SARIMAX(
            training,
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=60)
    fit_seconds = time.perf_counter() - started
    converged = bool(fit.mle_retvals.get("converged", False))
    predictions = []
    last_seen = cutoff
    for origin in origins:
        if origin > last_seen:
            new_observations = series.loc[last_seen + pd.Timedelta(hours=1) : origin]
            fit = fit.append(new_observations, refit=False)
            last_seen = origin
        predictions.append(np.asarray(fit.forecast(steps=24), dtype=float))
    return np.stack(predictions), fit_seconds, converged


def _add_prediction(
    records: list[pd.DataFrame],
    test: pd.DataFrame,
    client: str,
    fold: int,
    model: str,
    values: np.ndarray,
    scale_mean: float,
    scale_mase: float,
    peak_threshold: float,
) -> None:
    result = test[["origin", "target_time", "horizon", "actual_kw"]].copy()
    result["client"] = client
    result["fold"] = fold
    result["model"] = model
    result["predicted_kw"] = np.asarray(values, dtype=float).reshape(-1)
    result["train_mean_kw"] = scale_mean
    result["train_seasonal_mae_kw"] = scale_mase
    result["train_peak_threshold_kw"] = peak_threshold
    records.append(result)


def evaluate_client(series: pd.Series, client: str, folds: list[tuple[str, str]], include_sarima: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    tabular, sequences = build_samples(series)
    if tabular.empty:
        raise ValueError(f"{client} 没有足够的训练样本。")
    prediction_records: list[pd.DataFrame] = []
    runtime_rows = []
    for fold_number, (start_text, end_text) in enumerate(folds, start=1):
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        # Every training label, including the +24h target, must be known at cutoff.
        train = tabular.loc[tabular["target_time"] <= start]
        test = tabular.loc[tabular["origin"].between(start, end)].copy()
        train_sequences = sequences.loc[sequences["origin"].isin(train["origin"].unique())]
        train_sequences = train_sequences.loc[
            train_sequences["origin"] + pd.Timedelta(hours=24) <= start
        ]
        test_sequences = sequences.loc[sequences["origin"].between(start, end)]
        if len(test) != 14 * 24 or len(test_sequences) != 14:
            raise ValueError(f"{client} 第{fold_number}折的测试样本不足14天。")
        training_hourly = series.loc[:start]
        mean = float(training_hourly.mean())
        std = max(float(training_hourly.std()), 1e-6)
        seasonal_mae = max(float(np.mean(np.abs(training_hourly.to_numpy()[24:] - training_hourly.to_numpy()[:-24]))), 1e-6)
        peak_threshold = float(training_hourly.quantile(0.9))
        print(f"{client} fold {fold_number}: train={len(train)} rows, test={len(test)} rows", flush=True)

        _add_prediction(prediction_records, test, client, fold_number, "yesterday", test["target_lag_24h"].to_numpy(), mean, seasonal_mae, peak_threshold)
        _add_prediction(prediction_records, test, client, fold_number, "last_week", test["target_lag_168h"].to_numpy(), mean, seasonal_mae, peak_threshold)

        started = time.perf_counter()
        ridge_lags = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        ridge_lags.fit(train[TABULAR_LAG_FEATURES], train["actual_kw"])
        fit_seconds = time.perf_counter() - started
        predicted = ridge_lags.predict(test[TABULAR_LAG_FEATURES])
        _add_prediction(prediction_records, test, client, fold_number, "ridge_lags", predicted, mean, seasonal_mae, peak_threshold)
        runtime_rows.append({"client": client, "fold": fold_number, "model": "ridge_lags", "fit_seconds": fit_seconds, "converged": True})

        started = time.perf_counter()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
        ridge.fit(train[TABULAR_FEATURES], train["actual_kw"])
        fit_seconds = time.perf_counter() - started
        predicted = ridge.predict(test[TABULAR_FEATURES])
        _add_prediction(prediction_records, test, client, fold_number, "ridge", predicted, mean, seasonal_mae, peak_threshold)
        runtime_rows.append({"client": client, "fold": fold_number, "model": "ridge", "fit_seconds": fit_seconds, "converged": True})

        started = time.perf_counter()
        tree = HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=12, l2_regularization=1.0, random_state=42,
        )
        tree.fit(train[TABULAR_FEATURES], train["actual_kw"])
        fit_seconds = time.perf_counter() - started
        predicted = tree.predict(test[TABULAR_FEATURES])
        _add_prediction(prediction_records, test, client, fold_number, "hist_gb", predicted, mean, seasonal_mae, peak_threshold)
        runtime_rows.append({"client": client, "fold": fold_number, "model": "hist_gb", "fit_seconds": fit_seconds, "converged": True})

        started = time.perf_counter()
        direct_linear = fit_direct_linear(train_sequences, mean, std)
        fit_seconds = time.perf_counter() - started
        predicted = predict_direct_linear(direct_linear, test_sequences, mean, std)
        _add_prediction(prediction_records, test, client, fold_number, "direct_linear", predicted, mean, seasonal_mae, peak_threshold)
        runtime_rows.append({"client": client, "fold": fold_number, "model": "direct_linear", "fit_seconds": fit_seconds, "converged": True})

        started = time.perf_counter()
        dlinear = fit_dlinear(train_sequences, mean, std)
        fit_seconds = time.perf_counter() - started
        predicted = predict_dlinear(dlinear, test_sequences, mean, std)
        _add_prediction(prediction_records, test, client, fold_number, "dlinear", predicted, mean, seasonal_mae, peak_threshold)
        runtime_rows.append({"client": client, "fold": fold_number, "model": "dlinear", "fit_seconds": fit_seconds, "converged": True})

        if include_sarima:
            origins = sorted(test_sequences["origin"].tolist())
            predicted, fit_seconds, converged = predict_sarima(series, start, origins)
            _add_prediction(prediction_records, test, client, fold_number, "sarima", predicted, mean, seasonal_mae, peak_threshold)
            runtime_rows.append({"client": client, "fold": fold_number, "model": "sarima", "fit_seconds": fit_seconds, "converged": converged})
        print(f"{client} fold {fold_number}: completed", flush=True)

    return pd.concat(prediction_records, ignore_index=True), pd.DataFrame(runtime_rows)


def metric_table(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in predictions.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        error = group["actual_kw"].to_numpy() - group["predicted_kw"].to_numpy()
        peak = group["actual_kw"] >= group["train_peak_threshold_kw"]
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "samples": len(group),
                "mae_kw": float(np.mean(np.abs(error))),
                "rmse_kw": float(np.sqrt(np.mean(error**2))),
                "nmae_percent": float(np.mean(np.abs(error) / group["train_mean_kw"]) * 100),
                "mase": float(np.mean(np.abs(error) / group["train_seasonal_mae_kw"])),
                "peak_mae_kw": float(np.mean(np.abs(error[peak.to_numpy()]))) if peak.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_plots(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    clients = sorted(metrics["client"].unique())
    fig, axes = plt.subplots(1, len(clients), figsize=(5 * len(clients), 5), squeeze=False)
    for ax, client in zip(axes[0], clients, strict=True):
        rows = metrics.loc[metrics["client"] == client].set_index("model")
        models = [name for name in MODEL_NAMES if name in rows.index]
        ax.bar(models, rows.loc[models, "nmae_percent"], color="steelblue")
        ax.set(title=client, ylabel="nMAE (% of training mean)")
        ax.tick_params(axis="x", labelrotation=60)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Real UCI load benchmark: model comparison by client")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    for client in sorted(predictions["client"].unique()):
        sample = predictions.loc[
            (predictions["client"] == client)
            & (predictions["fold"] == predictions["fold"].max())
            & (predictions["target_time"] >= pd.Timestamp("2014-09-22"))
        ]
        fig, ax = plt.subplots(figsize=(13, 5))
        actual = sample.loc[sample["model"] == "yesterday"]
        ax.plot(actual["target_time"], actual["actual_kw"], label="Actual", linewidth=1.8)
        for name in ["yesterday", "ridge", "hist_gb", "direct_linear", "dlinear", "sarima"]:
            current = sample.loc[sample["model"] == name]
            if not current.empty:
                ax.plot(current["target_time"], current["predicted_kw"], label=name, alpha=0.85)
        ax.set(title=f"{client}: final 7-day forecasts", xlabel="Target time", ylabel="Load (kW)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"forecast_{client}.png", dpi=150)
        plt.close(fig)


def save_report(metrics: pd.DataFrame, runtime: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# UCI 真实负荷预测实验报告",
        "",
        "三位用户、三个14天回测窗口。预测起点为午夜，预测未来1～24小时；所有模型使用相同测试点。",
        "",
        "MAE/RMSE 单位为 kW；nMAE = MAE / 每折训练集平均负荷；MASE 的分母为训练集昨日同期基线平均绝对误差。",
        "",
    ]
    for client in sorted(metrics["client"].unique()):
        rows = metrics.loc[metrics["client"] == client].sort_values("nmae_percent")
        lines.extend(
            [
                f"## {client}",
                "",
                "| 模型 | MAE (kW) | RMSE (kW) | nMAE (%) | MASE | 高负荷时段 MAE (kW) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows.itertuples():
            lines.append(
                f"| {row.model} | {row.mae_kw:.2f} | {row.rmse_kw:.2f} | "
                f"{row.nmae_percent:.2f} | {row.mase:.3f} | {row.peak_mae_kw:.2f} |"
            )
        lines.extend(["", f"该用户按 nMAE 排名最好的模型：**{rows.iloc[0]['model']}**。", ""])
    if "sarima" in runtime["model"].values:
        failed = runtime.loc[(runtime["model"] == "sarima") & (~runtime["converged"])]
        lines.append(f"SARIMA 未收敛折次：{len(failed)}。")
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "- 数据只有负荷，不含真实的天气预报；本实验不能说明气象特征的收益。",
            "- 只选取3位用户和2014年4～9月窗口，结论不能直接推广到全部370位用户或全年。",
            "- DLinear 风格模型是趋势/残差分解后的线性多输出模型，使用岭回归训练，未逐字复现论文的优化过程。",
            "- 表格模型使用目标日历信息，序列模型仅使用历史负荷；模型排名同时反映输入特征和模型结构差异。ridge_lags 与 ridge、direct_linear 与 dlinear 是控制输入的消融对照。",
            "- SARIMA 阶数固定，Ridge、树和 DLinear 超参数亦固定；没有使用测试集选模型或调参。",
            "- 逐折与逐提前量明细见 metrics_by_fold.csv 和 metrics_by_horizon.csv。",
            "",
        ]
    )
    (output_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="真实负荷的回归与时序预测对照实验")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clients", nargs="+", help="只评估指定用户；默认评估准备数据中的全部用户")
    parser.add_argument("--folds", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--skip-sarima", action="store_true", help="快速调试时跳过 SARIMA")
    args = parser.parse_args()
    data = load_hourly(args.data.resolve())
    clients = args.clients or data.columns.tolist()
    if not set(clients).issubset(data.columns):
        raise ValueError("指定用户不在已准备的数据中。")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_list = []
    runtimes_list = []
    for client in clients:
        predictions, runtimes = evaluate_client(data[client], client, FOLDS[: args.folds], not args.skip_sarima)
        predictions_list.append(predictions)
        runtimes_list.append(runtimes)
    predictions = pd.concat(predictions_list, ignore_index=True)
    runtime = pd.concat(runtimes_list, ignore_index=True)
    overall = metric_table(predictions, ["client", "model"])
    horizons = metric_table(predictions, ["client", "model", "horizon"])
    per_fold = metric_table(predictions, ["client", "fold", "model"])
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    overall.to_csv(output_dir / "metrics_by_client.csv", index=False)
    horizons.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    per_fold.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    runtime.to_csv(output_dir / "fit_runtime.csv", index=False)
    save_plots(predictions, overall, output_dir)
    save_report(overall, runtime, output_dir)
    summary = {
        "data": str(args.data.resolve()),
        "clients": clients,
        "folds": FOLDS[: args.folds],
        "models": [name for name in MODEL_NAMES if name != "sarima" or not args.skip_sarima],
        "forecast_definition": "At midnight after observing origin load, forecast hours +1 through +24",
        "training_rule": "Only target labels available at each fold start are used for fitting",
        "tabular_features": TABULAR_FEATURES,
        "tabular_lags_only_features": TABULAR_LAG_FEATURES,
        "dlinear_note": "DLinear-style moving-average decomposition with two linear weight blocks, fitted jointly by L2-regularized least squares",
        "weather_note": "UCI source has load only; no future weather is used",
        "source_period": [str(data.index.min()), str(data.index.max())],
        "test_prediction_rows": len(predictions),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(overall.sort_values(["client", "nmae_percent"]).to_string(index=False), flush=True)
    print(f"结果保存在 {output_dir}", flush=True)


if __name__ == "__main__":
    main()
