"""Streamlit demonstration for the stage 2 and stage 3 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE2_DIR = PROJECT_ROOT / "outputs" / "stage2"
DEMO_REQUEST = PROJECT_ROOT / "outputs" / "stage3" / "demo_forecast_request.json"
DATA_PATH = PROJECT_ROOT / "data" / "demo_load.csv"

st.set_page_config(page_title="园区负荷预测", page_icon="⚡", layout="wide")
st.title("园区负荷预测与异常监测")
st.caption("第二阶段回测结果 + 第三阶段预测服务。数据和异常注入均为演示用途。")


def call_api(url: str, endpoint: str, payload: dict) -> dict | None:
    try:
        response = requests.post(f"{url.rstrip('/')}{endpoint}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
        st.error(f"接口调用失败：{detail}")
        return None


with st.sidebar:
    st.header("服务连接")
    api_url = st.text_input("API 地址", "http://127.0.0.1:8000")
    try:
        health = requests.get(f"{api_url.rstrip('/')}/health", timeout=3).json()
        if health.get("status") == "ok":
            st.success(f"已连接；模型训练截止：{health['trained_through']}")
        else:
            st.warning(health.get("detail", "模型尚未就绪"))
    except requests.RequestException:
        st.warning("API 未启动。先运行 README 中的 uvicorn 命令。")


backtest_tab, forecast_tab, anomaly_tab = st.tabs(["滚动回测", "24小时预测", "异常监测"])

with backtest_tab:
    summary_path = STAGE2_DIR / "backtest_summary.json"
    predictions_path = STAGE2_DIR / "backtest_predictions.csv"
    if not summary_path.exists() or not predictions_path.exists():
        st.info("先运行 python src/run_stage2.py 生成回测结果。")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path, parse_dates=["target_time"])
        left, middle, right = st.columns(3)
        left.metric("模型 MAE", f"{summary['model']['mae_kw']:.2f} kW")
        middle.metric("模型 MAPE", f"{summary['model']['mape_percent']:.2f}%")
        right.metric("优于基线的提前量", f"{summary['horizons_beating_baseline']}/24")
        st.subheader("最近7天：实际负荷与预测负荷")
        recent = predictions.tail(7 * 24).set_index("target_time")
        st.line_chart(
            recent[["actual_load_kw", "predicted_load_kw", "baseline_load_kw"]].rename(
                columns={
                    "actual_load_kw": "实际负荷",
                    "predicted_load_kw": "模型预测",
                    "baseline_load_kw": "昨日同期",
                }
            )
        )
        horizon_path = STAGE2_DIR / "horizon_metrics.csv"
        if horizon_path.exists():
            st.subheader("未来第1～24小时的误差")
            by_horizon = pd.read_csv(horizon_path).set_index("horizon")
            st.line_chart(
                by_horizon[["model_mae_kw", "baseline_mae_kw"]].rename(
                    columns={"model_mae_kw": "模型 MAE", "baseline_mae_kw": "基线 MAE"}
                )
            )
        segment_path = STAGE2_DIR / "segment_metrics.csv"
        if segment_path.exists():
            st.subheader("时段分析")
            st.dataframe(pd.read_csv(segment_path), hide_index=True, use_container_width=True)
        st.caption("回测按时间滚动；上图来自已保存的测试期结果，不会重新训练。")

with forecast_tab:
    st.write("接口输入：预测起点之前连续168小时负荷，以及未来24小时逐时天气预报。输出：24个目标小时的预测负荷。")
    uploaded = st.file_uploader("可上传符合接口格式的预测请求 JSON", type="json", key="forecast_upload")
    request_payload = None
    if uploaded is not None:
        try:
            request_payload = json.load(uploaded)
        except json.JSONDecodeError as exc:
            st.error(f"JSON 格式错误：{exc}")
    elif DEMO_REQUEST.exists():
        request_payload = json.loads(DEMO_REQUEST.read_text(encoding="utf-8"))
        st.info("当前使用留出日期的历史回放请求；演示天气采用目标时刻的实测温度。")
    else:
        st.info("先运行 python src/prepare_stage3.py 生成演示请求。")

    if request_payload is not None:
        st.caption(f"预测起点：{request_payload.get('forecast_origin', '未知')}")
        if st.button("运行未来24小时预测", type="primary"):
            result = call_api(api_url, "/forecast", request_payload)
            if result is not None:
                st.session_state["forecast_result"] = result
                st.session_state["forecast_request"] = request_payload

    result = st.session_state.get("forecast_result")
    if result is not None:
        frame = pd.DataFrame(result["predictions"])
        frame["target_time"] = pd.to_datetime(frame["target_time"])
        st.subheader("预测结果")
        st.line_chart(
            frame.set_index("target_time")[["predicted_load_kw", "previous_day_baseline_kw"]].rename(
                columns={"predicted_load_kw": "模型预测", "previous_day_baseline_kw": "昨日同期"}
            )
        )
        st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button(
            "下载预测 CSV",
            data=frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="day_ahead_forecast.csv",
            mime="text/csv",
        )

with anomaly_tab:
    st.write("取得目标小时的实际负荷后，接口用实际值与预测值的残差计算异常分数。")
    active_request = st.session_state.get("forecast_request")
    if active_request is None:
        st.info("先在“24小时预测”页运行一次预测。")
    elif not DATA_PATH.exists():
        st.info("演示观测数据不存在；请从 API /anomalies 提交实际负荷。")
    else:
        raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).set_index("timestamp")
        origin = pd.Timestamp(active_request["forecast_origin"])
        target_times = [origin + pd.Timedelta(hours=h) for h in range(1, 25)]
        if not set(target_times).issubset(raw.index):
            st.info("当前请求不在演示数据中；请通过 API /anomalies 提交对应目标时刻的实际负荷。")
        else:
            injection = st.selectbox("演示异常", ["不注入", "第10小时负荷突增 40%", "第10小时负荷突降 40%"])
            if st.button("检测这24小时的观测值"):
                observations = []
                for horizon, target in enumerate(target_times, start=1):
                    load = float(raw.loc[target, "load_kw"])
                    if horizon == 10 and injection != "不注入":
                        load *= 1.4 if "突增" in injection else 0.6
                    observations.append({"timestamp": target.isoformat(), "load_kw": round(load, 2)})
                response = call_api(
                    api_url,
                    "/anomalies",
                    {"forecast": active_request, "observations": observations},
                )
                if response is not None:
                    st.session_state["anomaly_result"] = response
            anomaly_result = st.session_state.get("anomaly_result")
            if anomaly_result is not None:
                scored = pd.DataFrame(anomaly_result["results"])
                scored["target_time"] = pd.to_datetime(scored["target_time"])
                st.metric("高风险及严重告警数", anomaly_result["alert_count"])
                st.line_chart(
                    scored.set_index("target_time")[["observed_load_kw", "predicted_load_kw"]].rename(
                        columns={"observed_load_kw": "实际负荷", "predicted_load_kw": "预测负荷"}
                    )
                )
                st.dataframe(scored.loc[scored["is_alert"]], hide_index=True, use_container_width=True)
                st.caption("注入选项仅修改本次演示观测值；历史回测和模型文件不会变化。")
