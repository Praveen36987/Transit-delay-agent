"""
Unit & Integration Test Suite for Transit Delay Agent
"""

import os
from pathlib import Path
import duckdb
import pandas as pd
import pytest

from database import get_db_connection
from analytics import (
    DEFAULT_DB_PATH,
    detect_anomalies,
    get_route_baselines,
    get_weather_impact_summary,
    get_worst_delayed_routes,
    predict_route_delay,
)
from evaluate import (
    evaluate_anomaly_detection,
    evaluate_delay_prediction_mae,
    temporal_train_test_split,
    verify_tool_call_grounding,
)
from ingest import fetch_weather, init_db, seed_synthetic_data
from mcp_server import (
    get_active_disruption_anomalies,
    get_route_hourly_baselines,
    get_weather_delay_correlation,
    get_worst_delayed_routes as mcp_worst_routes,
    predict_delay_with_weather,
)


@pytest.fixture(scope="session")
def test_db(tmp_path_factory):
    """Create a temporary test DuckDB database populated with seeded records."""
    temp_dir = tmp_path_factory.mktemp("test_data")
    db_path = temp_dir / "test_transit_logs.duckdb"
    seed_synthetic_data(db_path=db_path, num_records=200, days=7)
    return db_path


def test_database_initialization(test_db):
    """Verify table creation and record counts."""
    con = get_db_connection(test_db)
    count = con.execute("SELECT COUNT(*) FROM trip_delays").fetchone()[0]
    assert count == 200


def test_weather_fetch():
    """Verify weather fetch returns float tuple."""
    precip, wind = fetch_weather(17.3850, 78.4867)
    assert isinstance(precip, float)
    assert isinstance(wind, float)
    assert precip >= 0.0
    assert wind >= 0.0


def test_analytics_anomaly_detection(test_db):
    """Verify Isolation Forest flags anomalies."""
    anomalies = detect_anomalies(db_path=test_db, contamination=0.05)
    assert not anomalies.empty
    assert "anomaly_flag" in anomalies.columns
    assert "anomaly_score" in anomalies.columns
    assert (anomalies["anomaly_flag"] == -1).all()


def test_route_baselines(test_db):
    """Verify route delay baseline aggregations."""
    df = get_route_baselines(db_path=test_db, route_id="ROUTE_101")
    assert not df.empty
    assert "avg_delay_min" in df.columns
    assert "p90_delay_min" in df.columns


def test_worst_delayed_routes(test_db):
    """Verify worst delayed routes query."""
    df = get_worst_delayed_routes(db_path=test_db, hours_back=168, limit=3)
    assert len(df) <= 3
    assert "avg_delay_min" in df.columns


def test_weather_delay_correlation(test_db):
    """Verify weather impact summary extraction."""
    summary = get_weather_impact_summary(db_path=test_db)
    assert "Normal Weather" in summary or "Adverse Weather" in summary


def test_predict_route_delay(test_db):
    """Verify delay prediction function."""
    pred = predict_route_delay("ROUTE_101", hour_of_day=8, precipitation_mm=20.0, db_path=test_db)
    assert pred["route_id"] == "ROUTE_101"
    assert pred["target_hour"] == 8
    assert "predicted_delay_min" in pred
    assert pred["weather_penalty_min"] > 0


def test_mcp_tools():
    """Verify MCP tools format valid markdown/json strings."""
    worst = mcp_worst_routes(hours_back=168, limit=3)
    assert isinstance(worst, str)
    assert len(worst) > 0

    anomalies = get_active_disruption_anomalies(top_n=3)
    assert isinstance(anomalies, str)

    pred_json = predict_delay_with_weather("ROUTE_101", 9, precipitation_mm=0.0)
    assert "predicted_delay_min" in pred_json


def test_grounding_validator():
    """Verify that hallucinated route IDs or numbers trigger review."""
    valid_tool_output = "Route METRO_RED has avg delay 12.5 mins. ROUTE_101 has avg delay 5.2 mins."

    # Grounded response
    grounded_llm = "METRO_RED is currently experiencing an average delay of 12.5 minutes."
    res = verify_tool_call_grounding(grounded_llm, valid_tool_output)
    assert res["is_strictly_grounded"] is True
    assert len(res["unverified_routes"]) == 0

    # Ungrounded response with hallucinated route
    hallucinated_llm = "ROUTE_999 is facing extreme 99.9 minute delays."
    res_fake = verify_tool_call_grounding(hallucinated_llm, valid_tool_output, valid_routes=["ROUTE_999"])
    assert res_fake["is_strictly_grounded"] is False
    assert "ROUTE_999" in res_fake["unverified_routes"]
