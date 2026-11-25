"""
Analytics & Anomaly Detection Module
Provides deterministic analytics, route delay baselines, and Scikit-learn
IsolationForest anomaly detection over DuckDB transit logs.
Uses a centralized database connection to prevent configuration mismatch errors.
"""

from pathlib import Path
from typing import Optional
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from database import get_db_connection

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "transit_logs.duckdb"
STATIC_ROUTES_PATH = BASE_DIR / "data" / "static" / "routes.txt"


def load_static_routes() -> pd.DataFrame:
    """Load human-readable route information if available."""
    if STATIC_ROUTES_PATH.exists():
        try:
            return pd.read_csv(STATIC_ROUTES_PATH)
        except Exception:
            pass
    return pd.DataFrame()


def detect_anomalies(
    db_path: Path | str = DEFAULT_DB_PATH,
    contamination: float = 0.05,
    days_back: int = 7,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Train Scikit-learn IsolationForest on multi-feature delay context:
    [hour_of_day, arrival_delay_sec, precipitation_mm, wind_speed_kmh]
    Flags abnormal cascading delays rather than regular schedule jitter.
    """
    if con is None:
        con = get_db_connection(db_path)

    query = """
    SELECT
        captured_at,
        trip_id,
        route_id,
        stop_id,
        EXTRACT(HOUR FROM captured_at) AS hour_of_day,
        arrival_delay_sec,
        ROUND(arrival_delay_sec / 60.0, 2) AS delay_minutes,
        precipitation_mm,
        wind_speed_kmh
    FROM trip_delays
    WHERE captured_at >= (SELECT MAX(captured_at) FROM trip_delays) - (INTERVAL '1 day' * ?)
    """
    df = con.execute(query, [days_back]).df()

    if len(df) < 50:
        return pd.DataFrame()

    features = df[["hour_of_day", "arrival_delay_sec", "precipitation_mm", "wind_speed_kmh"]].copy()

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    df["anomaly_flag"] = model.fit_predict(features)  # -1 = anomaly, 1 = normal
    df["anomaly_score"] = model.decision_function(features)  # lower = more abnormal

    anomalies = df[df["anomaly_flag"] == -1].sort_values("arrival_delay_sec", ascending=False)

    routes_df = load_static_routes()
    if not routes_df.empty and "route_id" in routes_df.columns:
        anomalies = anomalies.merge(routes_df[["route_id", "route_long_name"]], on="route_id", how="left")

    return anomalies


def get_route_baselines(
    db_path: Path | str = DEFAULT_DB_PATH,
    route_id: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Compute route-level delay baselines:
    - Average delay (min)
    - 90th percentile delay (p90 min)
    - Total recorded trips
    Grouped by route_id and hour_of_day.
    """
    if con is None:
        con = get_db_connection(db_path)

    filter_sql = "WHERE route_id = ?" if route_id else ""
    params = [route_id] if route_id else []

    query = f"""
    SELECT
        route_id,
        EXTRACT(HOUR FROM captured_at) AS hour_of_day,
        COUNT(*) AS snapshot_count,
        ROUND(AVG(arrival_delay_sec) / 60.0, 2) AS avg_delay_min,
        ROUND(QUANTILE_CONT(arrival_delay_sec, 0.90) / 60.0, 2) AS p90_delay_min,
        ROUND(MAX(arrival_delay_sec) / 60.0, 2) AS max_delay_min
    FROM trip_delays
    {filter_sql}
    GROUP BY route_id, EXTRACT(HOUR FROM captured_at)
    ORDER BY route_id, hour_of_day
    """
    df = con.execute(query, params).df()
    return df


def get_worst_delayed_routes(
    db_path: Path | str = DEFAULT_DB_PATH,
    hours_back: int = 48,
    limit: int = 5,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Return routes with highest average delay over recent hours.
    """
    if con is None:
        con = get_db_connection(db_path)

    query = """
    SELECT
        route_id,
        COUNT(*) AS total_updates,
        ROUND(AVG(arrival_delay_sec) / 60.0, 2) AS avg_delay_min,
        ROUND(QUANTILE_CONT(arrival_delay_sec, 0.90) / 60.0, 2) AS p90_delay_min,
        ROUND(MAX(arrival_delay_sec) / 60.0, 2) AS max_delay_min,
        ROUND(AVG(precipitation_mm), 2) AS avg_rain_mm
    FROM trip_delays
    WHERE captured_at >= (SELECT MAX(captured_at) FROM trip_delays) - (INTERVAL '1 hour' * ?)
    GROUP BY route_id
    ORDER BY avg_delay_min DESC
    LIMIT ?
    """
    df = con.execute(query, [hours_back, limit]).df()

    routes_df = load_static_routes()
    if not routes_df.empty and "route_id" in routes_df.columns:
        df = df.merge(routes_df[["route_id", "route_long_name"]], on="route_id", how="left")

    return df


def get_weather_impact_summary(
    db_path: Path | str = DEFAULT_DB_PATH,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> dict:
    """
    Analyze the statistical correlation between adverse weather (rain > 5mm or wind > 25km/h)
    and transit delays.
    """
    if con is None:
        con = get_db_connection(db_path)

    res = con.execute("""
        SELECT
            CASE
                WHEN precipitation_mm > 5.0 OR wind_speed_kmh > 25.0 THEN 'Adverse Weather'
                ELSE 'Normal Weather'
            END AS condition,
            COUNT(*) AS count,
            ROUND(AVG(arrival_delay_sec) / 60.0, 2) AS avg_delay_min,
            ROUND(QUANTILE_CONT(arrival_delay_sec, 0.90) / 60.0, 2) AS p90_delay_min
        FROM trip_delays
        GROUP BY 1
    """).fetchall()

    summary = {row[0]: {"count": row[1], "avg_delay_min": row[2], "p90_delay_min": row[3]} for row in res}
    return summary


def predict_route_delay(
    route_id: str,
    hour_of_day: int,
    precipitation_mm: float = 0.0,
    wind_speed_kmh: float = 10.0,
    db_path: Path | str = DEFAULT_DB_PATH,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> dict:
    """
    Deterministic predictive baseline calculation combining historical route-hour
    quantiles and weather sensitivity regressions.
    """
    if con is None:
        con = get_db_connection(db_path)

    row = con.execute("""
        SELECT
            ROUND(AVG(arrival_delay_sec) / 60.0, 2) AS baseline_avg,
            ROUND(QUANTILE_CONT(arrival_delay_sec, 0.90) / 60.0, 2) AS baseline_p90,
            COUNT(*) AS sample_size
        FROM trip_delays
        WHERE route_id = ? AND EXTRACT(HOUR FROM captured_at) = ?
    """, [route_id, hour_of_day]).fetchone()

    baseline_avg = float(row[0]) if (row and row[0] is not None) else 5.0
    baseline_p90 = float(row[1]) if (row and row[1] is not None) else 15.0
    sample_size = int(row[2]) if row else 0

    # Weather impact factor
    weather_penalty = 0.0
    if precipitation_mm > 10.0:
        weather_penalty += (precipitation_mm - 10.0) * 0.45
    if wind_speed_kmh > 30.0:
        weather_penalty += (wind_speed_kmh - 30.0) * 0.25

    predicted_delay_min = round(baseline_avg + weather_penalty, 2)
    risk_level = "High" if predicted_delay_min >= 20.0 else ("Moderate" if predicted_delay_min >= 10.0 else "Low")

    return {
        "route_id": route_id,
        "target_hour": hour_of_day,
        "sample_size": sample_size,
        "historical_avg_delay_min": baseline_avg,
        "historical_p90_delay_min": baseline_p90,
        "weather_penalty_min": round(weather_penalty, 2),
        "predicted_delay_min": predicted_delay_min,
        "risk_level": risk_level,
    }


if __name__ == "__main__":
    print("--- Worst Delayed Routes (Recent) ---")
    print(get_worst_delayed_routes())
    print("\n--- Active Disruption Anomalies (IsolationForest) ---")
    anomalies = detect_anomalies().head(5)
    print(anomalies[["captured_at", "route_id", "stop_id", "delay_minutes", "precipitation_mm", "anomaly_score"]])
    print("\n--- Weather Impact Analysis ---")
    print(get_weather_impact_summary())
    print("\n--- Delay Prediction for ROUTE_101 at Hour 9 (Rain 25mm) ---")
    print(predict_route_delay("ROUTE_101", 9, precipitation_mm=25.0, wind_speed_kmh=35.0))
