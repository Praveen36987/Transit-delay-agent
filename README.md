# Local Edge-AI Logistics & Transit Delay Prediction Agent

A 100% free, local edge-AI agent that ingests public transit (GTFS-RT) and weather (Open-Meteo) feeds, persists delay logs in an embedded DuckDB database, trains Scikit-learn `IsolationForest` models for unsupervised disruption detection, and connects a local LLM (Ollama / Gemma 4 / Qwen 3.5) to local analytics tools using the Model Context Protocol (MCP) and Pydantic AI.

---

## 1. System Architecture (4 Local Layers)

```
┌────────────────────────────────────────────────────────┐
│  Agent Orchestration Layer                             │
│  Local Ollama LLM (Gemma 4, Qwen 3.5) via Pydantic AI  │
└───────────────────────────▲────────────────────────────┘
                            │ Tool Calling via Stdio
┌───────────────────────────▼────────────────────────────┐
│  Tool Interface Layer (Local FastMCP Server)          │
│  stdio-based MCP server exposing verified queries     │
└───────────────────────────▲────────────────────────────┘
                            │ Internal Python Calls
┌───────────────────────────▼────────────────────────────┐
│  Analytics & Anomaly Detection Layer                   │
│  DuckDB aggregations + Scikit-learn Isolation Forest   │
└───────────────────────────▲────────────────────────────┘
                            │ Persisted Snapshots
┌───────────────────────────▼────────────────────────────┐
│  Data Ingestion Layer                                  │
│  GTFS-RT Protobuf feeds + Open-Meteo weather API       │
└────────────────────────────────────────────────────────┘
```

1. **Data Ingestion Layer (`ingest.py`)**: Polls public GTFS Realtime feeds and correlates delay records with Open-Meteo weather conditions (precipitation, wind speed).
2. **Analytics & Storage Layer (`analytics.py`)**: Embedded columnar SQL queries in DuckDB + Scikit-learn `IsolationForest` trained on `[hour_of_day, arrival_delay_sec, precipitation_mm, wind_speed_kmh]`.
3. **Tool Interface Layer (`mcp_server.py`)**: FastMCP server exposing delay queries, baselines, and predictions over stdio (zero ports or network exposure).
4. **Agent Orchestration Layer (`agent.py`)**: Pydantic AI agent configured with local Ollama (`gemma4:e4b` / `qwen3.5`) with strict tool-grounding system prompts.
5. **Evaluation & Backtesting Loop (`evaluate.py`)**: Chronological 70/30 train-holdout backtest measuring disruption precision/recall, delay MAE, and tool-call hallucination checks.

---

## 2. Directory Structure

```text
transit-delay-agent/
├── data/
│   ├── static/                  # Static GTFS schedule context (routes.txt, stops.txt)
│   └── transit_logs.duckdb      # Embedded time-series delay & weather database
├── ingest.py                    # GTFS-RT + Open-Meteo poller & synthetic data seeder
├── analytics.py                 # DuckDB queries & Scikit-learn Isolation Forest
├── mcp_server.py                # Local stdio MCP server exposing analytics tools
├── agent.py                     # Pydantic AI agent wired to Ollama
├── evaluate.py                  # Weekly backtest script for MAE and alert precision
├── test_pipeline.py             # Pytest automated test suite
├── requirements.txt             # Python dependencies
└── README.md                    # Documentation & setup guide
```

---

## 3. Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Seed or Ingest Transit Data

Populate the local DuckDB database with 7 days of realistic transit snapshots:
```bash
python ingest.py --seed --records 600 --days 7
```

To poll a live GTFS-RT feed:
```bash
python ingest.py --once --url "YOUR_GTFS_RT_URL" --lat 17.3850 --lon 78.4867
```

Check database statistics anytime:
```bash
python ingest.py --status
```

---

## 4. Run Analytics & Anomaly Detection Directly

Inspect worst delayed routes, isolation forest anomaly flags, and weather correlation:
```bash
python analytics.py
```

Sample output:
- **Worst Delayed Routes**: Table showing average delay, p90 delay, max delay, and rain correlation per route.
- **Active Disruption Anomalies**: Isolation Forest flags anomalous cascading delays with `anomaly_score < 0`.
- **Weather Impact**: Quantitative breakdown comparing normal weather vs adverse weather delays.

---

## 5. Launch the Local FastMCP Server

The MCP server runs over stdio and can be integrated into Antigravity, Claude Desktop, or custom MCP clients:
```bash
python mcp_server.py
```

### Exposed Tools:
* `get_worst_delayed_routes(hours_back, limit)`: Returns top delayed transit corridors.
* `get_active_disruption_anomalies(contamination, top_n)`: Returns recent statistical anomalies.
* `get_route_hourly_baselines(route_id)`: 24-hour historical p50/p90 baseline profile.
* `get_weather_delay_correlation()`: Performance metrics split by weather condition.
* `predict_delay_with_weather(route_id, hour_of_day, precipitation_mm, wind_speed_kmh)`: Deterministic delay prediction.

---

## 6. Run the Local LLM Agent

### Running with Ollama:
1. Start Ollama:
   ```bash
   ollama serve
   ```
2. Pull your chosen lightweight model:
   ```bash
   ollama pull gemma4:e4b
   # Or: ollama pull qwen2.5:3b
   ```
3. Run the agent:
   ```bash
   python agent.py --query "Which routes are currently experiencing abnormal delays, and is weather a factor?"
   ```

### Interactive Query Session:
```bash
python agent.py --interactive
```

### Offline / Dry-Run Verification Mode:
If running without a GPU or before starting Ollama, run in dry-run mode to verify tool execution and grounded reporting:
```bash
python agent.py --dry-run
```

---

## 7. Backtesting & Model Evaluation Loop

Run temporal holdout evaluation (train on Days 1–5, test on Days 6–7):
```bash
python evaluate.py --threshold 10.0 --contamination 0.05
```

This evaluates:
1. **Disruption Alert Precision & Recall**: Accuracy of Isolation Forest alerts against real sustained delays (>10 min).
2. **Mean Absolute Error (MAE)**: Compares baseline and weather-adjusted predictive delay against holdout ground truth.
3. **Tool-Call Grounding Check**: Cross-checks every route ID and delay number cited by the LLM against raw tool outputs to ensure zero hallucinations.

---

## 8. Run Automated Test Suite

Run the full pytest suite:
```bash
pytest test_pipeline.py -v
```
All 9 unit and integration tests verify the end-to-end pipeline from database initialization to MCP server output formatting and grounding checks.
