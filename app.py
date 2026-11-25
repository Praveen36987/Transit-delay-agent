"""
Streamlit Web Application for Local Edge-AI Logistics & Transit Delay Prediction Agent
Provides an interactive dashboard for DuckDB delay logs, Scikit-learn IsolationForest
disruption anomalies, weather correlations, what-if delay predictions, and an AI chat assistant.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local analytics imports
from analytics import (
    DEFAULT_DB_PATH,
    detect_anomalies,
    get_route_baselines,
    get_weather_impact_summary,
    get_worst_delayed_routes,
    predict_route_delay,
)
from ingest import fetch_weather, init_db, seed_synthetic_data

# Page configuration
st.set_page_config(
    page_title="Edge-AI Transit Delay Agent",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    """Ensure database exists and return connection."""
    init_db(DEFAULT_DB_PATH)
    return duckdb.connect(str(DEFAULT_DB_PATH), read_only=False)


def check_ollama_status(base_url="http://localhost:11434"):
    """Check if local Ollama daemon is active."""
    import requests
    try:
        r = requests.get(base_url, timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


# ---------------- SIDEBAR ----------------
st.sidebar.title("🚆 Transit Edge-AI")
st.sidebar.caption("Zero-cost local delay analytics & LLM reasoning")

con = get_db()
count = con.execute("SELECT COUNT(*) FROM trip_delays").fetchone()[0]

st.sidebar.markdown(f"**Database Status:**")
st.sidebar.info(f"📁 Records in DuckDB: **{count:,}**")

if count == 0:
    st.sidebar.warning("No records found! Click below to seed demo logs.")

if st.sidebar.button("🎲 Seed 500 Historical Records"):
    with st.spinner("Seeding realistic delay records..."):
        seed_synthetic_data(DEFAULT_DB_PATH, num_records=500, days=7)
        st.sidebar.success("Successfully seeded 500 records!")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Local LLM (Ollama)")
ollama_online = check_ollama_status()
if ollama_online:
    st.sidebar.success("🟢 Ollama Daemon: Online")
else:
    st.sidebar.warning("🟡 Ollama Daemon: Offline (Using deterministic grounded mode)")

selected_model = st.sidebar.selectbox(
    "Agent Model",
    ["gemma4:e4b", "qwen2.5:3b", "llama3.2:3b", "qwen3.5:latest"],
    index=0,
)

# Fetch current live weather
lat, lon = 17.3850, 78.4867
live_precip, live_wind = fetch_weather(lat, lon)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Live Weather ({lat}, {lon})**")
st.sidebar.write(f"🌧️ Precipitation: `{live_precip:.1f} mm`")
st.sidebar.write(f"💨 Wind Speed: `{live_wind:.1f} km/h`")


# ---------------- MAIN APP ----------------
st.title("Local Edge-AI Logistics & Delay Prediction Agent")
st.markdown(
    "Combines **DuckDB**, **Scikit-learn Isolation Forest**, and **local LLM reasoning via MCP** "
    "to analyze transit bottlenecks and predict cascading disruptions without paid APIs."
)

if count == 0:
    st.warning("Your local DuckDB database is currently empty. Use the sidebar button **'Seed 500 Historical Records'** to get started.")
    st.stop()

# Top KPI metrics
earliest, latest = con.execute("SELECT MIN(captured_at), MAX(captured_at) FROM trip_delays").fetchone()
avg_delay = con.execute("SELECT ROUND(AVG(arrival_delay_sec)/60.0, 2) FROM trip_delays").fetchone()[0]
p90_delay = con.execute("SELECT ROUND(QUANTILE_CONT(arrival_delay_sec, 0.90)/60.0, 2) FROM trip_delays").fetchone()[0]
unique_routes = con.execute("SELECT COUNT(DISTINCT route_id) FROM trip_delays").fetchone()[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Logged Trips", f"{count:,}")
col2.metric("Tracked Corridors", f"{unique_routes}")
col3.metric("Fleet Avg Delay", f"{avg_delay} min")
col4.metric("Fleet p90 Delay", f"{p90_delay} min")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Delay Baselines & Corridors",
    "🚨 Anomaly Detection (Isolation Forest)",
    "🌦️ Weather Impact Analysis",
    "🤖 Agent Assistant & Predictor",
])

# ---------------- TAB 1: BASELINES ----------------
with tab1:
    st.subheader("Route-Level Delay Baselines & Bottlenecks")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        hours_filter = st.slider("Lookback Window (Hours)", min_value=6, max_value=168, value=48, step=6)
        worst_df = get_worst_delayed_routes(hours_back=hours_filter, limit=10)
        
        if not worst_df.empty:
            fig_bar = px.bar(
                worst_df,
                x="route_id",
                y="avg_delay_min",
                color="avg_delay_min",
                title=f"Top Delayed Routes (Past {hours_filter} Hours)",
                labels={"avg_delay_min": "Average Delay (minutes)", "route_id": "Route ID"},
                color_continuous_scale="Reds",
                hover_data=["route_long_name", "p90_delay_min", "max_delay_min"],
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(worst_df, use_container_width=True)
    
    with col_right:
        routes_list = con.execute("SELECT DISTINCT route_id FROM trip_delays ORDER BY 1").df()["route_id"].tolist()
        selected_route = st.selectbox("Inspect Hourly Baseline for Route:", routes_list)
        
        baseline_df = get_route_baselines(route_id=selected_route)
        if not baseline_df.empty:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=baseline_df["hour_of_day"],
                y=baseline_df["avg_delay_min"],
                mode="lines+markers",
                name="Average Delay",
                line=dict(color="#1976D2", width=2),
            ))
            fig_line.add_trace(go.Scatter(
                x=baseline_df["hour_of_day"],
                y=baseline_df["p90_delay_min"],
                mode="lines+markers",
                name="90th Percentile (p90)",
                line=dict(color="#D32F2F", width=2, dash="dot"),
            ))
            fig_line.update_layout(
                title=f"24-Hour Diurnal Delay Curve — {selected_route}",
                xaxis_title="Hour of Day (24h)",
                yaxis_title="Delay (Minutes)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_line, use_container_width=True)


# ---------------- TAB 2: ANOMALY DETECTION ----------------
with tab2:
    st.subheader("Unsupervised Cascading Disruption Detection")
    st.markdown(
        "Trains a Scikit-learn **`IsolationForest`** on multi-dimensional features "
        "`[hour_of_day, arrival_delay_sec, precipitation_mm, wind_speed_kmh]` to separate routine jitter from severe disruptions."
    )
    
    col_c1, col_c2 = st.columns([1, 3])
    with col_c1:
        contamination = st.slider("Outlier Contamination Threshold", 0.01, 0.15, 0.05, 0.01)
        top_anomalies = st.number_input("Max Anomalies to Display", 5, 50, 15)
        
    anomalies_df = detect_anomalies(contamination=contamination)
    
    if not anomalies_df.empty:
        with col_c2:
            st.metric("Flagged Outliers", f"{len(anomalies_df)} anomalies", delta="Isolation Forest Flagged", delta_color="inverse")
            
        fig_scatter = px.scatter(
            anomalies_df,
            x="precipitation_mm",
            y="delay_minutes",
            color="anomaly_score",
            size="delay_minutes",
            hover_data=["captured_at", "route_id", "stop_id"],
            title="Disruption Anomalies: Delay vs. Precipitation",
            labels={"delay_minutes": "Arrival Delay (min)", "precipitation_mm": "Precipitation (mm)"},
            color_continuous_scale="Plasma",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("### Most Critical Anomaly Logs")
        st.dataframe(anomalies_df.head(top_anomalies), use_container_width=True)
    else:
        st.info("No anomalies detected or insufficient records.")


# ---------------- TAB 3: WEATHER IMPACT ----------------
with tab3:
    st.subheader("Weather Sensitivity & Correlation Analysis")
    weather_summary = get_weather_impact_summary()
    
    c_w1, c_w2 = st.columns(2)
    with c_w1:
        st.json(weather_summary)
        
    with c_w2:
        labels = list(weather_summary.keys())
        avg_delays = [weather_summary[k]["avg_delay_min"] for k in labels]
        p90_delays = [weather_summary[k]["p90_delay_min"] for k in labels]
        
        fig_w = go.Figure(data=[
            go.Bar(name='Average Delay (min)', x=labels, y=avg_delays, marker_color='#0288D1'),
            go.Bar(name='p90 Delay (min)', x=labels, y=p90_delays, marker_color='#C2185B')
        ])
        fig_w.update_layout(barmode='group', title="Delay Inflation: Normal vs. Adverse Weather")
        st.plotly_chart(fig_w, use_container_width=True)


# ---------------- TAB 4: AGENT ASSISTANT & PREDICTOR ----------------
with tab4:
    st.subheader("Transit Analyst Agent & Predictive Delay Engine")
    
    col_pred, col_chat = st.columns([1, 1])
    
    with col_pred:
        st.markdown("#### 🔮 What-If Delay Predictor")
        st.caption("Calculates grounded delay estimates combining route baselines & weather penalties.")
        
        p_route = st.selectbox("Predict for Route:", routes_list, key="pred_route")
        p_hour = st.slider("Target Hour of Day", 0, 23, datetime.now().hour)
        p_rain = st.number_input("Forecasted Rain (mm)", 0.0, 100.0, float(live_precip))
        p_wind = st.number_input("Forecasted Wind (km/h)", 0.0, 120.0, float(live_wind))
        
        if st.button("Calculate Delay Forecast"):
            pred_res = predict_route_delay(p_route, p_hour, p_rain, p_wind)
            st.success(f"**Predicted Delay: {pred_res['predicted_delay_min']} minutes**")
            st.write(f"- Historical Route Average: `{pred_res['historical_avg_delay_min']} min`")
            st.write(f"- Weather Penalty: `+{pred_res['weather_penalty_min']} min`")
            st.write(f"- Disruption Risk Level: **{pred_res['risk_level']}**")
    
    with col_chat:
        st.markdown("#### 💬 Ask the AI Transit Analyst")
        st.caption("Queries local MCP tools and DuckDB logs without hallucinations.")
        
        user_query = st.text_input(
            "Query the agent:",
            placeholder="Which routes are currently experiencing abnormal delays, and is weather a factor?",
        )
        
        if st.button("Run Agent Analysis"):
            if not user_query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Agent querying local MCP tools and synthesizing report..."):
                    if ollama_online:
                        try:
                            # Invoke agent asynchronously
                            from agent import build_agent
                            agent_instance, _ = build_agent(model_name=selected_model)
                            
                            async def run_chat():
                                async with agent_instance:
                                    res = await agent_instance.run(user_query)
                                    return res.output
                            
                            output = asyncio.run(run_chat())
                            st.markdown("### 🤖 Agent Grounded Response:")
                            st.markdown(output)
                        except Exception as e:
                            st.error(f"Error calling local Ollama agent: {e}")
                            st.info("Falling back to deterministic grounded report:")
                            from mcp_server import get_active_disruption_anomalies, get_weather_delay_correlation, get_worst_delayed_routes
                            st.markdown("#### Active Worst Delayed Routes:")
                            st.markdown(get_worst_delayed_routes(limit=3))
                            st.markdown("#### Flagged Disruption Anomalies:")
                            st.markdown(get_active_disruption_anomalies())
                    else:
                        st.info("Local Ollama is currently offline. Generating deterministic MCP-grounded summary:")
                        from mcp_server import get_active_disruption_anomalies, get_weather_delay_correlation, get_worst_delayed_routes
                        st.markdown("#### 1. Real Delay Logs (MCP Tool: `get_worst_delayed_routes`):")
                        st.markdown(get_worst_delayed_routes(limit=3))
                        st.markdown("#### 2. Isolation Forest Outliers (MCP Tool: `get_active_disruption_anomalies`):")
                        st.markdown(get_active_disruption_anomalies())
                        st.markdown("#### 3. Weather Correlation:")
                        st.json(get_weather_delay_correlation())
