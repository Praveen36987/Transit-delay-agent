"""
Centralized Database Connection Manager for DuckDB
Maintains a shared connection per database file with consistent configuration (read_only=False),
preventing configuration mismatch errors between Streamlit, analytics, and ingestion.
"""

from pathlib import Path
from typing import Dict, Optional
import duckdb

_CONNECTION_CACHE: Dict[str, duckdb.DuckDBPyConnection] = {}


def get_db_connection(
    db_path: Path | str,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """
    Get or create a cached DuckDB connection for the specified path.
    Always uses read_only=False so reads, writes, and Streamlit caching share
    the exact same connection instance without configuration conflicts.
    """
    path_str = str(Path(db_path).resolve())
    conn = _CONNECTION_CACHE.get(path_str)
    
    # Check if connection exists and is still open
    if conn is None:
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(path_str, read_only=False)
        _CONNECTION_CACHE[path_str] = conn

    return conn


def close_db_connection(db_path: Path | str):
    """Close and remove a specific cached connection."""
    global _CONNECTION_CACHE
    path_str = str(Path(db_path).resolve())
    if path_str in _CONNECTION_CACHE:
        try:
            _CONNECTION_CACHE[path_str].close()
        except Exception:
            pass
        del _CONNECTION_CACHE[path_str]


def close_all_connections():
    """Close all cached connections."""
    global _CONNECTION_CACHE
    for path, conn in list(_CONNECTION_CACHE.items()):
        try:
            conn.close()
        except Exception:
            pass
    _CONNECTION_CACHE.clear()
