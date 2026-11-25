"""
Local MCP Server for Transit Delay Analytics
Exposes DuckDB queries, route baselines, and Scikit-learn anomaly detection
as standardized tools over stdio for local LLM reasoning.
"""

import json
from pathlib import Path

# Dual compatibility for MCP 1.x and MCP 2.x
try:
    from mcp.server.mcpserver import MCPServer
    FastMCP = MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from fastmcp import FastMCP

import duckdb
from analytics import (
    DEFAULT_DB_PATH,
    detect_anomalies,
    get_route_baselines,
    get_weather_impact_summary,
    get_worst_delayed_routes as analytics_worst_delayed_routes,
    predict_route_delay,
)


def safe_to_markdown(df) -> str:
    """Format DataFrame as markdown table with graceful fallback to to_string."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


mcp = FastMCP("transit-delay-analytics")


@mcp.tool()
def get_worst_delayed_routes(hours_back: int = 48, limit: int = 5) -> str:
    """
    Return transit routes with the highest average arrival delays in the specified recent time window.
    Args:
        hours_back: Number of past hours to inspect (default: 48).
        limit: Maximum number of routes to return (default: 5).
    """
    df = analytics_worst_delayed_routes(hours_back=hours_back, limit=limit)
    if df.empty:
        return "No recent transit update logs found in the selected time window."
    cols = [c for c in ["route_id", "route_long_name", "total_updates", "avg_delay_min", "p90_delay_min", "max_delay_min", "avg_rain_mm"] if c in df.columns]
    return safe_to_markdown(df[cols])


@mcp.tool()
def get_active_disruption_anomalies(contamination: float = 0.05, top_n: int = 10) -> str:
    """
    Return recent statistical delay anomalies flagged by Scikit-learn IsolationForest
    over multi-dimensional transit & weather features [hour, delay, precipitation, wind].
    Args:
        contamination: Contamination proportion of outliers to flag (default: 0.05).
        top_n: Number of most abnormal entries to return (default: 10).
    """
    anomalies = detect_anomalies(contamination=contamination).head(top_n)
    if anomalies.empty:
        return "No anomalies detected or insufficient historical data (<50 records)."

    cols = [c for c in ["captured_at", "route_id", "route_long_name", "stop_id", "delay_minutes", "precipitation_mm", "wind_speed_kmh", "anomaly_score"] if c in anomalies.columns]
    return safe_to_markdown(anomalies[cols])


@mcp.tool()
def get_route_hourly_baselines(route_id: str) -> str:
    """
    Return 24-hour historical delay baselines (average delay, 90th percentile delay, max delay)
    for a specific transit route.
    Args:
        route_id: Target transit route identifier (e.g. 'ROUTE_101', 'METRO_RED').
    """
    df = get_route_baselines(route_id=route_id)
    if df.empty:
        return f"No baseline data found for route '{route_id}'."
    return safe_to_markdown(df)


@mcp.tool()
def get_weather_delay_correlation() -> str:
    """
    Return comparative transit performance metrics under normal weather versus adverse weather (rain > 5mm or wind > 25km/h).
    """
    summary = get_weather_impact_summary()
    return json.dumps(summary, indent=2)


@mcp.tool()
def predict_delay_with_weather(
    route_id: str,
    hour_of_day: int,
    precipitation_mm: float = 0.0,
    wind_speed_kmh: float = 10.0,
) -> str:
    """
    Calculate deterministic delay prediction for a route based on target hour-of-day
    and forecasted weather conditions.
    Args:
        route_id: Route identifier (e.g., 'ROUTE_101', 'ROUTE_202', 'METRO_BLUE').
        hour_of_day: Hour in 24h format (0 to 23).
        precipitation_mm: Forecasted rainfall in mm (default: 0.0).
        wind_speed_kmh: Forecasted wind speed in km/h (default: 10.0).
    """
    pred = predict_route_delay(
        route_id=route_id,
        hour_of_day=hour_of_day,
        precipitation_mm=precipitation_mm,
        wind_speed_kmh=wind_speed_kmh,
    )
    return json.dumps(pred, indent=2)


if __name__ == "__main__":
    # Run the MCP server over stdio
    mcp.run()
