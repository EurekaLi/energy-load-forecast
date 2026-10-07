# 电力负荷预测与异常检测

一个面向能源算法岗位的可复现练习项目：从小时级负荷数据出发，完成日前 24 小时预测、按时间滚动回测、预测残差异常检测，以及 HTTP 接口和交互式演示；另以 UCI 真实负荷数据构建回归与时序预测的对照实验。

> **项目边界：** 真实 UCI 数据用于**预测模型比较**；异常检测与服务演示使用生成的园区数据和人工注入异常。项目未接入真实天气预报、真实异常标签或生产环境，演示指标不能解释为生产效果。

## 已实现功能

| 模块 | 主要内容 | 输入 → 输出 |
|---|---|---|
| 一步预测 | 梯度提升树与昨日同期基线、时间顺序测试 | 历史负荷／日历／温度 → 下一小时负荷 |
| 日前预测 | 24 个按预测提前量独立训练的模型 | 截至午夜的负荷历史与未来温度 → 未来 1～24 小时负荷 |
| 滚动回测 | 28 天校准、28 天测试，默认每 7 天重新拟合 | 历史小时序列 → 总体、分提前量和分时段误差 |
| 异常检测 | 按提前量的残差中位数与 MAD 校准，告警分级 | 预测值与到达的实测值 → 异常分数和事件 |
| 服务与页面 | FastAPI 接口、Streamlit 回测与预测展示 | JSON 请求 → 预测、告警和可视化 |
| 真实数据基准 | UCI 数据上比较季节性基线、Ridge、树、线性序列模型、DLinear 风格模型和 SARIMA | 负荷历史／可用日历 → 未来 24 小时负荷 |

## 数据与方法

### 演示数据：完整预测—告警链路

`src/generate_demo_data.py` 生成连续的小时级园区负荷、温度与日历数据。一步预测通过滞后值和先 `shift(1)` 再计算的滚动统计量避免读取目标时刻的负荷。日前预测每天午夜获取当前实际负荷，构建目标时刻的昨日／上周同期值、过去 24/168 小时统计量和时间特征；24 个 `HistGradientBoostingRegressor` 分别预测第 1～24 小时。

滚动回测中，每折仅使用截止时已经产生的训练标签。异常阈值来自**测试期之前**的校准残差：对每个预测提前量分别估计残差中位数及 MAD 稳健尺度，计算 `|残差 - 中位数| / 稳健尺度`，并按 2.5、3.5、5.0 三档阈值分级。演示脚本在预测完成后注入突增、突降和持续高负荷，仅用于检查检测链路。

演示数据中的未来温度使用目标时刻的**实测温度模拟完美预报**；接入真实业务时必须替换为预测发布时可取得的天气预报，不能把未来实测温度当作线上输入。

### UCI 真实数据：回归与时序模型比较

数据来自 [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)（Trindade, 2015；DOI: 10.24432/C58C86；CC BY 4.0）。从 370 个用电点的 15 分钟读数中截取 2014 年 4～9 月，以每小时 4 个 **kW 功率读数的平均值**得到连续的小时平均功率；不能直接求和后仍称为 kW。按非零率与负荷规模选取 `MT_140`、`MT_233`、`MT_273` 三位代表用户。该时间窗口避开原数据的 3 月、10 月夏令时特殊日。

每天午夜在观察到起点负荷后预测未来第 1～24 小时。所有方法使用同一批预测起点，分别在 2014-08-01～08-14、09-01～09-14、09-15～09-28 三个 14 天窗口回测；测试期间可使用新到达的历史负荷，但不更新模型参数。每位用户共有 1,008 个逐小时测试点。UCI 数据没有天气字段，因此这个实验**不使用未来天气**。

| 方法 | 单次模型输入 → 输出 | 比较目的 |
|---|---|---|
| 昨日／上周同期 | 目标时刻前 24/168 小时负荷 → 1 个值 | 无参数的周期性基线 |
| Ridge（仅历史） | 7 个滞后／统计特征 → 1 个值 | 日历特征消融 |
| Ridge | 12 个历史和日历特征 → 1 个值 | 带 L2 约束的线性回归 |
| HistGradientBoosting | 与 Ridge 相同的 12 个特征 → 1 个值 | 相同输入下比较非线性树模型 |
| 直接线性序列 | 过去 168 小时完整序列 → 未来 24 小时 | 多输出线性序列基线 |
| DLinear 风格 | 过去 168 小时的趋势与残差 → 未来 24 小时 | 比较移动平均分解；使用岭回归训练，**非原论文代码复现** |
| SARIMA | 截至起点的完整小时序列 → 未来 24 小时 | 固定阶数的传统季节性时序模型 |

表格模型的目标日历信息在预测时已知；所有负荷特征的时间戳均不晚于预测起点。不同类别模型的输入不完全相同，因此跨类别排名不能只归因于模型结构。详细设计见[真实数据实验说明](experiments/real_load_benchmark/README.md)。

## 实验结果

真实数据以 nMAE（逐折 MAE 除以该折训练平均负荷，越低越好）比较模型；同时保存 MAE、RMSE、MASE、峰值时段误差、逐折和逐预测提前量结果。

| 用户 | 昨日同期 | Ridge | 梯度提升树 | 直接线性序列 | DLinear 风格 | SARIMA | 最低 nMAE |
|---|---:|---:|---:|---:|---:|---:|---|
| MT_140 | **23.16%** | 34.99% | 27.58% | 41.98% | 42.15% | 34.81% | 昨日同期 |
| MT_233 | 5.28% | 4.80% | 5.04% | **4.36%** | 4.39% | 5.32% | 直接线性序列 |
| MT_273 | 5.46% | 5.20% | 5.47% | **5.17%** | 5.18% | 5.53% | 直接线性序列 |

MT_140 第三折的高负荷波动使昨日同期 nMAE 从前两折的 5.41%、6.67% 升至 57.41%，说明需要关注逐折稳定性。直接线性和 DLinear 风格模型在另两位用户上差距很小；趋势与残差之和仍是原序列，分解本身不保证提升表达能力。当前仅有三位用户和三个测试窗口，不能推断模型在全部用户或全年都保持相同排名。完整指标与图表可由下文命令生成至 `outputs/real_load_benchmark/`。

演示数据的日前预测滚动回测 MAE 为 **12.00 kW**，昨日同期基线为 **30.01 kW**，24 个预测提前量均优于该基线。人工强异常的检测结果仅是管线自检，不代表真实告警准确率。

## 安装与复现

需要 Python 3.10+。以下命令在 PowerShell 中从项目根目录执行；原始数据、处理后的数据和 `outputs/` 结果均被 `.gitignore` 排除，需要在本机自行生成。

```powershell
git clone https://github.com/EurekaLi/energy-load-forecast.git
cd energy-load-forecast
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**运行演示链路：**

```powershell
python src/generate_demo_data.py
python src/train_forecast.py
python src/run_stage2.py
python src/prepare_stage3.py
```

`outputs/` 将生成一步预测指标与图表、日前预测的滚动回测和异常检测结果，以及服务所需的 `forecast_bundle.joblib` 与演示请求。

**运行 UCI 真实数据基准：** 官方 ZIP 约 249 MiB；预处理直接从 ZIP 分块读取，无需展开完整文本。

```powershell
New-Item -ItemType Directory -Force data/uci_raw | Out-Null
curl.exe -L --retry 3 --fail --output data/uci_raw/electricityloaddiagrams20112014.zip https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
python experiments/real_load_benchmark/prepare_data.py
python experiments/real_load_benchmark/benchmark.py
```

输出包括 `metrics_by_client.csv`、`metrics_by_fold.csv`、`metrics_by_horizon.csv`、逐时预测、拟合耗时、对比图和 `experiment_report.md`。可用 `--folds 1` 先跑一折、`--clients MT_233` 指定用户、`--skip-sarima` 快速调试。

**启动接口与页面：** 先运行演示链路，再分别在两个终端启动：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
python -m streamlit run app/dashboard.py
```

接口文档位于 `http://127.0.0.1:8000/docs`，页面默认位于 `http://localhost:8501`。`POST /forecast` 接收午夜 `forecast_origin`、截至该时刻连续 168 小时的 `history`、未来连续 24 小时的 `future_weather`，返回 24 个目标时刻的预测值和昨日同期基线。`POST /anomalies` 再接收已到达的 1～24 个实际观测，返回残差、异常分数和告警级别；另有 `GET /health` 与 `GET /model-info`。具体请求字段和示例由 FastAPI 接口文档提供。

**接入自有小时级数据：** `src/train_forecast.py` 和 `src/run_stage2.py` 接收连续、无重复的 `timestamp,load_kw,temperature_c` CSV，例如：

```powershell
python src/train_forecast.py --data D:\path\to\hourly_load.csv --test-days 14
python src/run_stage2.py --data D:\path\to\hourly_load.csv
```

日前预测仍须保证目标时刻温度来自当时可用的**天气预报**，不能用未来实测值评估实际部署效果。

## 项目结构与测试

```text
energy-load-forecast/
├── src/                         # 演示数据、一步预测、日前回测、服务模型准备
├── experiments/real_load_benchmark/ # UCI 数据预处理与模型对照
├── api/                         # FastAPI 预测与异常接口
├── app/                         # Streamlit 演示页面
├── tests/                       # 时间对齐、特征、异常与接口测试
├── data/                        # 本地数据目录；数据文件不提交
├── outputs/                     # 本地模型与实验产物；结果不提交
└── requirements.txt
```

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 15 分钟到小时的单位处理、预测起点与标签对齐、历史特征不越界、训练与接口特征一致性、异常评分及错误输入校验。

## 后续方向

1. 接入真实负荷、预测发布时可用的天气预报、节假日及电价数据，评估天气预报误差对不同预测提前量的影响。
2. 扩展更多用户与季节，加入统计显著性和跨用户泛化评估，并研究概率预测、预测区间及高负荷时段的专门指标。
3. 收集并标注真实异常，按误报率、漏报率和告警时延重新校准阈值，替代人工注入异常的管线自检。
4. 增加数据漂移监测、定期重训与模型版本管理，验证长期运行稳定性。
