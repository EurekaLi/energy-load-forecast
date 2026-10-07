"""Prepare a small, reproducible hourly subset of the UCI load dataset."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
DEFAULT_ZIP = PROJECT_ROOT / "data" / "uci_raw" / "electricityloaddiagrams20112014.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "uci_hourly_selected.csv"


def read_window_from_zip(path: Path, start: str, end: str) -> pd.DataFrame:
    """Read chunks from the official ZIP without expanding its 678 MB text file."""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"不是完整的 ZIP 文件：{path}")
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    chunks = []
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if Path(name).name.lower() == "ld2011_2014.txt"]
        if len(members) != 1:
            raise ValueError(f"ZIP 中应有一个 LD2011_2014.txt，实际找到：{members}")
        with archive.open(members[0]) as source:
            reader = pd.read_csv(source, sep=";", decimal=",", chunksize=12000)
            for chunk in reader:
                timestamp_column = chunk.columns[0]
                chunk = chunk.rename(columns={timestamp_column: "timestamp"})
                chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="raise")
                window = chunk.loc[chunk["timestamp"].between(start_time, end_time)]
                if not window.empty:
                    chunks.append(window)
    if not chunks:
        raise ValueError("指定窗口内没有负荷数据。")
    frame = pd.concat(chunks, ignore_index=True).sort_values("timestamp")
    if frame["timestamp"].duplicated().any():
        raise ValueError("原始窗口存在重复时间戳。")
    return frame


def to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Average four 15-minute kW readings to hourly average power in kW."""
    indexed = frame.set_index("timestamp").sort_index()
    if not indexed.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("原始窗口不是连续的15分钟序列。")
    counts = indexed.iloc[:, 0].resample("h").size()
    if not counts.eq(4).all():
        raise ValueError("每小时必须有4条15分钟记录。")
    numeric = indexed.apply(pd.to_numeric, errors="raise")
    hourly = numeric.resample("h").mean()
    if hourly.isna().any().any():
        raise ValueError("小时聚合结果包含缺失值。")
    return hourly


def select_clients(hourly: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    if requested:
        missing = set(requested).difference(hourly.columns)
        if missing:
            raise ValueError(f"指定用户不存在：{sorted(missing)}")
        return requested
    candidates = []
    for client in hourly.columns:
        series = hourly[client]
        if (series > 0).mean() >= 0.98 and series.mean() > 1:
            candidates.append((client, float(series.mean())))
    if len(candidates) < 3:
        raise ValueError("满足非零率和平均负荷要求的用户不足3个。")
    candidates.sort(key=lambda item: (item[1], item[0]))
    indices = [round((len(candidates) - 1) * fraction) for fraction in (0.2, 0.5, 0.8)]
    return [candidates[index][0] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 UCI 真实负荷基准数据")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--start", default="2014-04-01 00:00:00")
    parser.add_argument("--end", default="2014-09-30 23:45:00")
    parser.add_argument("--clients", nargs="+", help="可指定 MT_001 等用户编号；默认按负荷规模选择3个")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = read_window_from_zip(args.zip.resolve(), args.start, args.end)
    hourly = to_hourly(raw)
    clients = select_clients(hourly, args.clients)
    selected = hourly[clients].reset_index()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False, float_format="%.6f")
    metadata = {
        "source": SOURCE_URL,
        "source_citation": "Trindade, A. (2015). ElectricityLoadDiagrams20112014. UCI. doi:10.24432/C58C86",
        "source_unit": "kW per 15-minute reading",
        "processed_unit": "hourly average power (kW)",
        "period": [str(selected["timestamp"].min()), str(selected["timestamp"].max())],
        "raw_rows_in_window": len(raw),
        "hourly_rows": len(selected),
        "clients": clients,
        "client_statistics": {
            client: {
                "mean_kw": round(float(hourly[client].mean()), 4),
                "nonzero_fraction": round(float((hourly[client] > 0).mean()), 4),
            }
            for client in clients
        },
        "selection": "20th, 50th and 80th load-scale percentile among clients with >=98% nonzero hours and mean >1 kW",
        "time_note": "April-September window avoids the dataset's March/October daylight-saving transition days",
    }
    with output.with_suffix(".metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
