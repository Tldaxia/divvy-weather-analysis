# Divvy 共享单车与天气分析

分析天气与共享单车骑行需求的关系。代码将芝加哥 Divvy 2023—2024 年行程与 NOAA 日度天气数据结合，提供数据下载、清洗聚合、用户类型比较、跨年验证和可视化。

## 环境要求

Python 3.14，依赖版本见 `divvy_weather/requirements.lock.txt`。

## 安装依赖

进入项目根目录后执行：

```bash
cd divvy_weather
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
mkdir -p data/raw data/processed results figures
```

后续命令均在 `divvy_weather/` 目录执行，使用 `.venv/bin/python` 调用已安装的依赖。

## 运行步骤

### 1. 下载数据

```bash
.venv/bin/python scripts/download.py
```

下载 2023、2024 年共 24 个月的 Divvy 行程文件、两处 NOAA 气象站数据和站点信息，保存到 `data/raw/`。`manifest.json` 记录来源地址、获取时间、大小和 SHA-256。

### 2. 清洗天气和 2023 年行程

```bash
.venv/bin/python scripts/prepare.py --weather --year 2023
```

处理后的天气 CSV、行程聚合 Parquet 和逐月质量检查 CSV 保存到 `data/processed/`。代码按分块方式读取行程文件，检查重复记录、时间、时长和用户类型。

### 3. 执行探索性分析

```bash
.venv/bin/python scripts/analyze.py discovery
```

使用 2023 年数据生成探索统计、模型估计和站点队列，数值输出在 `results/`，图像输出在 `figures/`。

### 4. 生成验证方案

```bash
.venv/bin/python scripts/freeze_protocol.py
```

首次运行时，在探索分析完成后、处理 2024 年行程前执行，生成 `refine-logs/protocol.json` 和哈希记录。方案生成后，后续运行直接从下一步继续。

### 5. 清洗 2024 年行程并验证

```bash
.venv/bin/python scripts/prepare.py --year 2024
.venv/bin/python scripts/analyze.py validation
```

验证阶段读取上一步在本地生成的 `refine-logs/protocol.json`，使用 2024 年数据执行回归、日期匹配、固定站点比较和跨年预测。

### 6. 执行车型敏感性分析

```bash
.venv/bin/python scripts/check_bicycle_only.py
```

剔除电动滑板车后重新计算用户行程比，输出 `results/bicycle_only_daily.csv`、`results/bicycle_only_effect.csv` 等文件。

## 代码结构

```text
divvy_weather/
├── requirements.lock.txt      # Python 依赖与版本
├── scripts/
│   ├── download.py            # 下载官方数据并记录来源
│   ├── prepare.py             # 清洗天气、行程与聚合数据
│   ├── analyze.py             # 探索分析与独立年份验证
│   ├── check_bicycle_only.py  # 剔除滑板车的敏感性分析
│   └── freeze_protocol.py     # 生成验证方案与哈希记录
└── README.md                 # 使用说明入口
```

运行产物按用途保存：`data/raw/` 存放下载数据，`data/processed/` 存放清洗与聚合数据，`results/` 存放数值结果，`figures/` 存放图像，`refine-logs/` 存放验证方案。以上目录均在本地生成。

## 数据来源

- [Divvy 官方行程数据](https://divvy-tripdata.s3.amazonaws.com/index.html)
- [NOAA GHCN-Daily 天气数据](https://www.ncei.noaa.gov/pub/data/ghcn/daily/)

