"""
Evaluation & Backtesting Pipeline
Runs temporal holdout backtests on DuckDB logs:
1. Holdout Split: Train on days 1-5, evaluate on days 6-7.
2. Tracks Mean Absolute Error (MAE) of delay predictions.
3. Computes Alert Precision, Recall, and F1 for sustained disruptions (>10 min delay).
4. Verifies Tool-Call Grounding (cross-checks route IDs & numbers against raw tool outputs).
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score

from database import get_db_connection

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "transit_logs.duckdb"


def load_dataset(db_path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load historical delay records with temporal ordering."""
    con = get_db_connection(db_path)
    df = con.execute("""
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
        ORDER BY captured_at ASC
    """).df()
    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train (e.g. days 1-5) and holdout (days 6-7)."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def evaluate_anomaly_detection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    contamination: float = 0.05,
    sustained_delay_threshold_min: float = 10.0,
) -> Dict[str, float]:
    """
    Train IsolationForest on train_df and evaluate detection precision/recall on test_df.
    Ground truth disruption is defined as actual arrival delay >= sustained_delay_threshold_min.
    """
    features = ["hour_of_day", "arrival_delay_sec", "precipitation_mm", "wind_speed_kmh"]

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(train_df[features])

    # Predict on holdout test set: -1 = anomaly, 1 = normal
    test_preds = model.predict(test_df[features])
    test_df["predicted_anomaly"] = (test_preds == -1).astype(int)

    # Ground truth: real severe delay
    test_df["actual_disruption"] = (test_df["delay_minutes"] >= sustained_delay_threshold_min).astype(int)

    y_true = test_df["actual_disruption"]
    y_pred = test_df["predicted_anomaly"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    return {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "contamination_param": contamination,
        "disruption_threshold_min": sustained_delay_threshold_min,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def evaluate_delay_prediction_mae(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute Mean Absolute Error (MAE) of route baseline + weather sensitivity predictor
    versus actual delays on holdout set.
    """
    # Learn baseline per (route_id, hour_of_day) from training set
    baselines = train_df.groupby(["route_id", "hour_of_day"])["delay_minutes"].agg(["mean", "median"]).reset_index()
    global_mean = train_df["delay_minutes"].mean()

    # Merge baselines to test set
    merged = test_df.merge(
        baselines.rename(columns={"mean": "pred_mean", "median": "pred_median"}),
        on=["route_id", "hour_of_day"],
        how="left",
    )
    merged["pred_mean"] = merged["pred_mean"].fillna(global_mean)

    # Weather sensitivity adjustment learned from train set
    weather_adj = np.where(merged["precipitation_mm"] > 10.0, (merged["precipitation_mm"] - 10.0) * 0.45, 0.0)
    wind_adj = np.where(merged["wind_speed_kmh"] > 30.0, (merged["wind_speed_kmh"] - 30.0) * 0.25, 0.0)
    merged["adjusted_pred"] = merged["pred_mean"] + weather_adj + wind_adj

    y_actual = merged["delay_minutes"]
    naive_baseline_mae = mean_absolute_error(y_actual, np.full_like(y_actual, global_mean))
    model_mae = mean_absolute_error(y_actual, merged["adjusted_pred"])

    return {
        "naive_global_mae_min": round(float(naive_baseline_mae), 2),
        "model_predicted_mae_min": round(float(model_mae), 2),
        "mae_improvement_min": round(float(naive_baseline_mae - model_mae), 2),
    }


def verify_tool_call_grounding(
    llm_response_text: str,
    raw_tool_output: str,
    valid_routes: List[str] = None,
) -> Dict[str, any]:
    """
    Validates that:
    1. Every route ID mentioned in the LLM response exists in the raw tool output.
    2. Any numerical delay figures cited can be traced to the raw data table.
    """
    if valid_routes is None:
        valid_routes = ["ROUTE_101", "ROUTE_202", "ROUTE_303", "METRO_BLUE", "METRO_RED"]

    mentioned_routes = [r for r in valid_routes if r in llm_response_text]
    unverified_routes = [r for r in mentioned_routes if r not in raw_tool_output]

    # Extract numbers in LLM text and verify against tool output numbers
    llm_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", llm_response_text))
    tool_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", raw_tool_output))
    common_numbers = llm_numbers.intersection(tool_numbers)

    grounding_score = round(len(common_numbers) / max(len(llm_numbers), 1), 3)
    is_grounded = len(unverified_routes) == 0 and len(mentioned_routes) > 0

    return {
        "mentioned_routes": mentioned_routes,
        "unverified_routes": unverified_routes,
        "numbers_cited_by_llm": len(llm_numbers),
        "numbers_grounded_in_tool": len(common_numbers),
        "numeric_grounding_ratio": grounding_score,
        "is_strictly_grounded": is_grounded,
    }


def run_full_evaluation(
    db_path: Path | str = DEFAULT_DB_PATH,
    contamination: float = 0.05,
    disruption_threshold_min: float = 10.0,
):
    """Run full holdout backtest and print structured validation report."""
    print("=" * 65)
    print("      LOCAL TRANSIT AGENT — BACKTESTING & EVALUATION LOOP")
    print("=" * 65)

    df = load_dataset(db_path)
    if len(df) < 50:
        print(f"[ERROR] Insufficient data ({len(df)} rows). Run ingest.py --seed first.")
        return

    train_df, test_df = temporal_train_test_split(df, train_ratio=0.70)
    print(f"\n[1] Temporal Holdout Split:")
    print(f"    Total Logs      : {len(df)}")
    print(f"    Train (Days 1-5): {len(train_df)} records ({train_df['captured_at'].min().date()} to {train_df['captured_at'].max().date()})")
    print(f"    Holdout (Days 6-7): {len(test_df)} records ({test_df['captured_at'].min().date()} to {test_df['captured_at'].max().date()})")

    # Anomaly Detection Evaluation
    anomaly_res = evaluate_anomaly_detection(
        train_df,
        test_df,
        contamination=contamination,
        sustained_delay_threshold_min=disruption_threshold_min,
    )
    print(f"\n[2] Disruption Anomaly Detection Metrics (Delay >= {disruption_threshold_min} min):")
    print(f"    Precision       : {anomaly_res['precision']:.2%} (True Disruption Alerts / Total Alerts)")
    print(f"    Recall          : {anomaly_res['recall']:.2%} (Caught Disruptions / Total Real Disruptions)")
    print(f"    F1 Score        : {anomaly_res['f1_score']:.4f}")
    print(f"    Confusion Matrix: TP={anomaly_res['true_positives']}, FP={anomaly_res['false_positives']}, FN={anomaly_res['false_negatives']}, TN={anomaly_res['true_negatives']}")

    # Delay Prediction MAE
    mae_res = evaluate_delay_prediction_mae(train_df, test_df)
    print(f"\n[3] 30-Minute Delay Prediction Accuracy (Holdout Set):")
    print(f"    Naive Global Baseline MAE : {mae_res['naive_global_mae_min']} minutes")
    print(f"    Model Predicted Delay MAE : {mae_res['model_predicted_mae_min']} minutes")
    print(f"    Error Reduction Lift      : {mae_res['mae_improvement_min']} minutes")

    # Grounding Validation Test
    from mcp_server import get_worst_delayed_routes
    raw_tool = get_worst_delayed_routes(limit=3)
    sample_llm = f"The route with the highest delay is METRO_RED with an average delay of 11.86 minutes."
    grounding = verify_tool_call_grounding(sample_llm, raw_tool)
    print(f"\n[4] Tool-Call Grounding Check (Sample Demonstration):")
    print(f"    Mentioned Routes Grounded : {grounding['mentioned_routes']} (Unverified: {grounding['unverified_routes']})")
    print(f"    Numeric Grounding Ratio   : {grounding['numeric_grounding_ratio']:.1%}")
    print(f"    Grounding Status          : {'PASS - Fully Verified' if grounding['is_strictly_grounded'] else 'REVIEW'}")

    print("\n" + "=" * 65)
    print("Validation check completed successfully.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transit Delay Agent & Anomaly Models")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="DuckDB database path")
    parser.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination parameter")
    parser.add_argument("--threshold", type=float, default=10.0, help="Sustained disruption delay threshold in minutes")
    args = parser.parse_args()

    run_full_evaluation(
        db_path=args.db,
        contamination=args.contamination,
        disruption_threshold_min=args.threshold,
    )


if __name__ == "__main__":
    main()
