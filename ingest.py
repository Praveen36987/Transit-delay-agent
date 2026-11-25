"""
GTFS-Realtime + Open-Meteo Ingestion Module
Ingests public transit delays and local weather conditions into a local DuckDB database.
Also includes a realistic synthetic seeder for zero-network testing and baseline generation.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import duckdb
import requests

try:
    from google.transit import gtfs_realtime_pb2
except ImportError:
    gtfs_realtime_pb2 = None

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "transit_logs.duckdb"

# Defaults
DEFAULT_LAT = 17.3850
DEFAULT_LON = 78.4867
DEFAULT_GTFS_RT_URL = os.environ.get(
    "GTFS_RT_URL",
    "https://api.mobilitydatabase.org/v1/feeds/mock/trip-updates"  # Can be replaced with actual city endpoint
)


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB table for trip delays and weather context."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS trip_delays (
            captured_at TIMESTAMP,
            trip_id VARCHAR,
            route_id VARCHAR,
            stop_id VARCHAR,
            arrival_delay_sec INTEGER,
            precipitation_mm DOUBLE,
            wind_speed_kmh DOUBLE
        )
    """)
    return con


def fetch_weather(lat: float, lon: float) -> Tuple[float, float]:
    """
    Fetch current weather (precipitation and wind speed) from Open-Meteo.
    Returns (precipitation_mm, wind_speed_kmh).
    Gracefully falls back if network is unreachable.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "precipitation,wind_speed_10m",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        return float(cur.get("precipitation", 0.0)), float(cur.get("wind_speed_10m", 0.0))
    except Exception as e:
        print(f"[WARN] Weather API call failed ({e}). Using baseline weather values.")
        return 0.0, 12.0


def poll_once(
    gtfs_rt_url: str = DEFAULT_GTFS_RT_URL,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """
    Poll GTFS Realtime protobuf endpoint once, correlate with weather, and append to DuckDB.
    """
    if gtfs_realtime_pb2 is None:
        raise RuntimeError("gtfs-realtime-bindings is not installed")

    precip, wind = fetch_weather(lat, lon)
    feed = gtfs_realtime_pb2.FeedMessage()

    print(f"Fetching GTFS-RT feed from: {gtfs_rt_url}")
    resp = requests.get(gtfs_rt_url, timeout=15)
    resp.raise_for_status()
    feed.ParseFromString(resp.content)

    now = datetime.now(timezone.utc)
    rows = []
    for entity in feed.entity:
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            trip_id = tu.trip.trip_id
            route_id = tu.trip.route_id
            for stu in tu.stop_time_update:
                if stu.HasField("arrival") and stu.arrival.HasField("delay"):
                    rows.append((
                        now,
                        trip_id,
                        route_id,
                        stu.stop_id,
                        int(stu.arrival.delay),
                        precip,
                        wind,
                    ))

    if rows:
        con = init_db(db_path)
        con.executemany("INSERT INTO trip_delays VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        con.close()
        print(f"[{now.isoformat()}] Logged {len(rows)} stop delay records into {db_path}.")
    else:
        print(f"[{now.isoformat()}] Feed parsed, but no stop_time_update arrival delays found.")

    return len(rows)


def seed_synthetic_data(
    db_path: Path | str = DEFAULT_DB_PATH,
    num_records: int = 500,
    days: int = 7,
    clear_existing: bool = False,
):
    """
    Generate realistic synthetic delay snapshots across past N days with:
    - Normal diurnal traffic delay variations (morning/evening peaks).
    - Weather impact (heavy rain -> cascading delays).
    - Injected disruption anomalies (extreme outliers) for Isolation Forest validation.
    """
    con = init_db(db_path)
    if clear_existing:
        con.execute("DELETE FROM trip_delays")
        print(f"Cleared existing records in {db_path}.")

    routes = ["ROUTE_101", "ROUTE_202", "ROUTE_303", "METRO_BLUE", "METRO_RED"]
    stops = {
        "ROUTE_101": ["STOP_SEC", "STOP_BEG", "STOP_PUN", "STOP_HIT"],
        "ROUTE_202": ["STOP_KOT", "STOP_MEH", "STOP_KUK", "STOP_PAT"],
        "ROUTE_303": ["STOP_DIL", "STOP_KOT", "STOP_PUN", "STOP_JUB"],
        "METRO_BLUE": ["STOP_SEC", "STOP_BEG", "STOP_HIT", "STOP_RAI"],
        "METRO_RED": ["STOP_MIY", "STOP_KUK", "STOP_PUN", "STOP_LBN"],
    }

    now = datetime.now(timezone.utc)
    rows = []
    random.seed(42)

    for i in range(num_records):
        # Sample timestamp randomly across the past 'days'
        time_offset = timedelta(
            days=random.uniform(0, days),
            hours=random.uniform(0, 24),
            minutes=random.uniform(0, 60),
        )
        record_time = now - time_offset
        hour = record_time.hour

        route = random.choice(routes)
        stop = random.choice(stops[route])
        trip_id = f"TRIP_{route}_{record_time.strftime('%Y%m%d%H%M')}_{random.randint(100, 999)}"

        # Diurnal pattern: rush hours (8-10 AM, 5-8 PM) have higher base delays
        is_rush = (8 <= hour <= 10) or (17 <= hour <= 20)
        base_delay = random.gauss(360, 120) if is_rush else random.gauss(60, 60)
        base_delay = max(0, int(base_delay))

        # Weather simulation
        # 15% chance of rain storm
        has_storm = random.random() < 0.15
        if has_storm:
            precip = round(random.uniform(15.0, 45.0), 2)
            wind = round(random.uniform(25.0, 55.0), 2)
            # Weather multiplier on delay
            base_delay = int(base_delay * random.uniform(1.8, 3.2) + 300)
        else:
            precip = round(random.uniform(0.0, 2.0), 2)
            wind = round(random.uniform(5.0, 18.0), 2)

        # Inject deliberate extreme anomalies (approx 5% of records)
        if random.random() < 0.05:
            # Major cascading delay (e.g. signal failure, track breakdown, waterlogging)
            base_delay += random.randint(1800, 4200)  # 30 - 70 mins delay

        rows.append((
            record_time,
            trip_id,
            route,
            stop,
            int(base_delay),
            float(precip),
            float(wind),
        ))

    con.executemany("INSERT INTO trip_delays VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    total_count = con.execute("SELECT COUNT(*) FROM trip_delays").fetchone()[0]
    con.close()
    print(f"Seeded {len(rows)} synthetic delay records into {db_path}. Total records: {total_count}.")


def print_status(db_path: Path | str = DEFAULT_DB_PATH):
    """Print current records count and time range from DuckDB."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Database {db_path} does not exist yet.")
        return
    con = duckdb.connect(str(db_path), read_only=True)
    count = con.execute("SELECT COUNT(*) FROM trip_delays").fetchone()[0]
    if count == 0:
        print("Database contains 0 records.")
        con.close()
        return

    earliest, latest = con.execute("SELECT MIN(captured_at), MAX(captured_at) FROM trip_delays").fetchone()
    avg_delay = con.execute("SELECT ROUND(AVG(arrival_delay_sec)/60.0, 2) FROM trip_delays").fetchone()[0]
    routes = con.execute("SELECT COUNT(DISTINCT route_id) FROM trip_delays").fetchone()[0]
    print(f"=== Transit Logs Status ({db_path}) ===")
    print(f"Total Records: {count}")
    print(f"Time Range   : {earliest} to {latest}")
    print(f"Routes Logged: {routes}")
    print(f"Overall Avg Delay: {avg_delay} mins")
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Transit & Weather Ingestion Engine")
    parser.add_argument("--seed", action="store_true", help="Generate synthetic realistic training records")
    parser.add_argument("--records", type=int, default=500, help="Number of records to seed (default: 500)")
    parser.add_argument("--days", type=int, default=7, help="Days of history to simulate (default: 7)")
    parser.add_argument("--clear", action="store_true", help="Clear existing records before seeding")
    parser.add_argument("--status", action="store_true", help="Show current database statistics")
    parser.add_argument("--once", action="store_true", help="Poll GTFS-RT feed once")
    parser.add_argument("--continuous", action="store_true", help="Continuously poll GTFS-RT feed")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (default: 300)")
    parser.add_argument("--url", type=str, default=DEFAULT_GTFS_RT_URL, help="GTFS-RT URL")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Path to DuckDB database")

    args = parser.parse_args()

    if args.seed:
        seed_synthetic_data(db_path=args.db, num_records=args.records, days=args.days, clear_existing=args.clear)
        print_status(db_path=args.db)
        return

    if args.status:
        print_status(db_path=args.db)
        return

    if args.once:
        poll_once(gtfs_rt_url=args.url, lat=args.lat, lon=args.lon, db_path=args.db)
        return

    if args.continuous:
        print(f"Starting continuous polling every {args.interval}s. Press Ctrl+C to stop.")
        while True:
            try:
                poll_once(gtfs_rt_url=args.url, lat=args.lat, lon=args.lon, db_path=args.db)
            except Exception as exc:
                print(f"[ERROR] Polling iteration failed: {exc}")
            time.sleep(args.interval)

    # Default action if no flags provided: seed demo data if empty and show status
    db_file = Path(args.db)
    if not db_file.exists() or db_file.stat().st_size == 0:
        print("Empty or new database detected. Seeding initial 500 historical records for quick testing...")
        seed_synthetic_data(db_path=args.db, num_records=500, days=7)
    print_status(db_path=args.db)


if __name__ == "__main__":
    main()
