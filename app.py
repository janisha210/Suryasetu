"""
app.py — SuryaSetu: Community Energy Resilience Platform
=========================================================
Streamlit application entry-point.

Run with:
    streamlit run app.py

All data displayed is SYNTHETIC and illustrative.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure src/ is on the import path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))
DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(DATA_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Internal modules ─────────────────────────────────────────────────────────
from data_loader import load_data, get_facility_columns, NUMERIC_COLUMNS
from validation import validate_dataframe, validate_scenario_params, validate_facility_config
from forecasting import ForecastPipeline
from essential_services import (
    DEFAULT_FACILITIES, Facility, evaluate_service_coverage, facility_from_dict,
    estimate_water_volume,
)
from weekly_planner import build_weekly_plan
from optimiser import compare_schedules
from scenarios import PRESET_SCENARIOS, run_scenario, build_custom_scenario
from metrics import energy_balance_summary, detect_anomalies, compute_coverage_stats
from reporting import export_forecast_csv, export_plan_csv, export_pdf, REPORTLAB_AVAILABLE

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SuryaSetu — Community Energy Resilience",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root palette ── */
:root {
    --clr-primary:   #1a7e5a;
    --clr-secondary: #f0b429;
    --clr-danger:    #c0392b;
    --clr-warning:   #e67e22;
    --clr-bg-card:   #f8fffe;
    --clr-border:    #cde8dc;
}

/* ── Enforce Clean White Page Background ── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}

/* ── Global Typography ── */
html, body, [class*="css"], .stMarkdown, p, span, label, div {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    color: #1f2937;
}

/* ── Headings with strong contrast on white ── */
h1, [data-testid="stHeader"] h1 {
    color: #0b3d2e !important;
    font-weight: 700;
}

h2 {
    color: #126346 !important;
    font-weight: 600;
    border-bottom: 2px solid #a3d9c2;
    padding-bottom: 4px;
}

h3 {
    color: #1a4f3b !important;
    font-weight: 600;
}

/* ── Sidebar: elegant light green theme with dark readable text ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f5ee 0%, #d5ece0 100%) !important;
    border-right: 1px solid #bce1ce !important;
}
[data-testid="stSidebar"] h1 {
    color: #0b3d2e !important;
    font-weight: 700;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #124d35 !important;
    font-weight: 600;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] div {
    color: #1f3d2e !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #0d3e2b !important;
    font-weight: 600 !important;
}
/* Input boxes inside sidebar: crisp white background with clear dark text and subtle borders */
[data-testid="stSidebar"] input {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border: 1px solid #9fcfb8 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] label span {
    color: #143828 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #badfcb !important;
}

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: #f4faf7;
    border: 1px solid #c0e6d6;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(13, 59, 46, 0.06);
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #124d35 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0b3323 !important;
    font-weight: 700 !important;
}

/* ── Warning & Alert Banners: explicitly dark text for readability ── */
.suryasetu-warn {
    background: #fff8e6;
    border-left: 5px solid #d97706;
    color: #78350f !important;
    font-weight: 500;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 10px 0;
}
.suryasetu-danger {
    background: #fee2e2;
    border-left: 5px solid #dc2626;
    color: #7f1d1d !important;
    font-weight: 500;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 10px 0;
}
.suryasetu-ok {
    background: #ecfdf5;
    border-left: 5px solid #059669;
    color: #064e3b !important;
    font-weight: 500;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 10px 0;
}
.suryasetu-info {
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    color: #1e3a8a !important;
    font-weight: 500;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 10px 0;
}
.synthetic-label {
    background: #fef3c7;
    border: 1px solid #d97706;
    padding: 5px 12px;
    border-radius: 5px;
    font-size: 0.82em;
    color: #92400e !important;
    font-weight: 600;
    display: inline-block;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_badge() -> None:
    st.markdown(
        '<span class="synthetic-label">⚠ Synthetic Data — Illustrative Only</span>',
        unsafe_allow_html=True,
    )


def danger_banner(msg: str) -> None:
    st.markdown(f'<div class="suryasetu-danger">🚨 {msg}</div>', unsafe_allow_html=True)


def warn_banner(msg: str) -> None:
    st.markdown(f'<div class="suryasetu-warn">⚠️ {msg}</div>', unsafe_allow_html=True)


def ok_banner(msg: str) -> None:
    st.markdown(f'<div class="suryasetu-ok">✅ {msg}</div>', unsafe_allow_html=True)


def info_banner(msg: str) -> None:
    st.markdown(f'<div class="suryasetu-info">ℹ️ {msg}</div>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def plotly_defaults() -> dict:
    return dict(
        template="plotly_white",
        color_discrete_sequence=["#1a7e5a", "#f0b429", "#c0392b", "#2980b9", "#8e44ad"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "raw_df": None,
        "pipeline": None,
        "facilities": None,
        "plan": None,
        "service_report": None,
        "scenario": None,
        "scenario_result": None,
        "comparison": None,
        "data_loaded": False,
        "pipeline_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & pipeline
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def _load_default_data() -> pd.DataFrame:
    return load_data()


def load_cached_data() -> pd.DataFrame:
    if "custom_dataset" in st.session_state and st.session_state["custom_dataset"] is not None:
        return st.session_state["custom_dataset"]
    return _load_default_data()


def get_pipeline(solar_scale: float = 1.0, demand_scale: float = 1.0) -> ForecastPipeline:
    raw = load_cached_data()
    # If custom dataset is present, build pipeline dynamically without cache key conflicts
    pipeline = ForecastPipeline(test_days=14)
    pipeline.run(raw, solar_scale=solar_scale, demand_scale=demand_scale)
    return pipeline


def get_facilities() -> list[Facility]:
    if st.session_state.get("facilities") is None:
        st.session_state["facilities"] = list(DEFAULT_FACILITIES)
    return st.session_state["facilities"]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("# ☀️ SuryaSetu")
        st.markdown("**Community Energy Resilience Platform**")
        st.markdown("---")

        st.markdown("### 🏘️ Community")
        community_name = st.text_input("Community Name", value="Illustrative Village")

        st.markdown("### 📊 Scenario Settings")
        solar_scale = st.slider(
            "Solar Scale", 0.1, 1.5, 1.0, 0.05,
            help="1.0 = normal; 0.7 = 30% reduced solar",
        )
        demand_scale = st.slider(
            "Demand Scale", 0.5, 2.0, 1.0, 0.05,
            help="1.0 = normal; 1.2 = 20% increased demand",
        )
        storage_kwh = st.number_input(
            "Battery Storage (kWh)", min_value=0.0, max_value=500.0,
            value=50.0, step=5.0,
        )
        grid_available = st.toggle("Grid Available", value=True)
        grid_capacity_kw = st.slider(
            "Grid Capacity (kW)", 0.0, 20.0, 5.0, 0.5,
            disabled=not grid_available,
        )

        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Page",
            [
                "🏠 Overview",
                "🏘️ Community Profile",
                "📈 Forecasts",
                "🔒 Essential Services",
                "📅 Weekly Plan",
                "🔬 Scenario Lab",
                "📄 Reports",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 📂 Data Source")
        uploaded_file = st.file_uploader(
            "Upload Community CSV",
            type=["csv"],
            help="Upload custom hourly energy data matching required schema (solar_generation_kw, demand columns, etc.)",
            key="custom_csv_uploader",
        )
        if uploaded_file is not None:
            try:
                import io
                content = uploaded_file.getvalue()
                uploaded_df = pd.read_csv(io.BytesIO(content), parse_dates=["timestamp"])
                vres = validate_dataframe(uploaded_df)
                if not vres.is_valid:
                    st.error("Uploaded CSV failed schema validation:")
                    for err in vres.errors:
                        st.caption(f"• {err}")
                else:
                    uploaded_df = uploaded_df.sort_values("timestamp").reset_index(drop=True)
                    uploaded_df = uploaded_df.set_index("timestamp")
                    for col in NUMERIC_COLUMNS:
                        if col in uploaded_df.columns:
                            uploaded_df[col] = pd.to_numeric(uploaded_df[col], errors="coerce")
                    st.session_state["custom_dataset"] = uploaded_df
                    st.success("✅ Custom data loaded!")
                    if st.button("Revert to Synthetic Data", key="revert_data_btn"):
                        st.session_state.pop("custom_dataset", None)
                        st.rerun()
            except Exception as ex:
                st.error(f"Error parsing uploaded CSV: {ex}")
        elif "custom_dataset" in st.session_state:
            st.info("Using uploaded custom dataset")
            if st.button("Revert to Synthetic Data", key="revert_data_btn2"):
                st.session_state.pop("custom_dataset", None)
                st.rerun()

        st.markdown("---")
        if "custom_dataset" in st.session_state:
            st.markdown('<span class="synthetic-label" style="background:#e0f2fe;border-color:#38bdf8;color:#0369a1!important;">📁 Custom Uploaded Dataset</span>', unsafe_allow_html=True)
        else:
            synthetic_badge()
        st.caption("v1.0 · Seva Innovation Challenge 2026")

    return {
        "page": page,
        "community_name": community_name,
        "solar_scale": solar_scale,
        "demand_scale": demand_scale,
        "storage_kwh": storage_kwh,
        "grid_available": grid_available,
        "grid_capacity_kw": grid_capacity_kw if grid_available else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Overview
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(ctx: dict) -> None:
    section_header(
        "🏠 Community Energy Overview",
        f"Community: {ctx['community_name']} · Scenario: Solar ×{ctx['solar_scale']}, "
        f"Demand ×{ctx['demand_scale']} · Storage: {ctx['storage_kwh']} kWh",
    )
    synthetic_badge()
    st.markdown("")

    pipeline = get_pipeline(ctx["solar_scale"], ctx["demand_scale"])
    fc = pipeline.forecast_df
    facilities = get_facilities()

    # Energy balance metrics
    balance = energy_balance_summary(fc, ctx["solar_scale"], ctx["demand_scale"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("☀️ Weekly Solar", f"{balance['total_solar_kwh']:.0f} kWh")
    c2.metric("⚡ Weekly Demand", f"{balance['total_demand_kwh']:.0f} kWh")
    c3.metric("📈 Net Surplus", f"{balance['net_surplus_kwh']:.0f} kWh")
    c4.metric("📉 Net Deficit", f"{balance['net_deficit_kwh']:.0f} kWh")
    c5.metric("🌿 Self-Sufficiency", f"{balance['self_sufficiency_pct']:.1f}%")

    st.markdown("---")

    # Weekly outlook chart
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📊 7-Day Energy Outlook")
        daily_solar = (fc["forecast_solar_kw"] * ctx["solar_scale"]).resample("D").sum()
        daily_demand = (fc["forecast_demand_kw"] * ctx["demand_scale"]).resample("D").sum()
        daily_df = pd.DataFrame({
            "Solar (kWh)": daily_solar,
            "Demand (kWh)": daily_demand,
        })
        fig = px.bar(
            daily_df,
            barmode="group",
            labels={"value": "Energy (kWh)", "variable": ""},
            color_discrete_map={"Solar (kWh)": "#1a7e5a", "Demand (kWh)": "#f0b429"},
            **{k: v for k, v in plotly_defaults().items() if k != "color_discrete_sequence"},
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🔒 Essential Services Status")
        service_report = evaluate_service_coverage(
            fc, facilities, ctx["solar_scale"], ctx["demand_scale"]
        )
        if service_report.has_critical_shortage:
            danger_banner(
                f"{len(service_report.critical_shortages)} critical shortage(s) detected."
            )
            for cs in service_report.critical_shortages[:3]:
                st.caption(f"• {cs.facility_label}: {cs.shortage_kwh:.1f} kWh unmet")
        else:
            ok_banner("All critical service requirements appear met.")

        # Coverage donut
        crit_facilities = [f for f in facilities if f.priority == "critical"]
        imp_facilities = [f for f in facilities if f.priority == "important"]
        flex_facilities = [f for f in facilities if f.priority == "flexible"]

        fig2 = go.Figure(data=[go.Pie(
            labels=["Critical", "Important", "Flexible"],
            values=[len(crit_facilities), len(imp_facilities), len(flex_facilities)],
            hole=0.55,
            marker_colors=["#c0392b", "#f0b429", "#1a7e5a"],
        )])
        fig2.update_layout(
            height=200, margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True, legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Forecast shortages / warnings
    st.markdown("### ⚠️ Forecast Warnings")
    plan = build_weekly_plan(
        fc, facilities,
        solar_scale=ctx["solar_scale"],
        demand_scale=ctx["demand_scale"],
        grid_capacity_kw=ctx["grid_capacity_kw"],
        storage_kwh=ctx["storage_kwh"],
        grid_available=ctx["grid_available"],
    )

    real_warnings = [w for w in plan.shortage_warnings if w.strip()]
    if real_warnings:
        for w in real_warnings:
            warn_banner(w)
    else:
        ok_banner("No forecast shortage warnings for this scenario.")

    if not ctx["grid_available"]:
        info_banner("Grid outage scenario active — all energy from solar + storage.")

    # Key metrics table
    st.markdown("### 📋 Key Modelled Metrics")
    info_banner(
        f"Renewable fraction: {plan.renewable_fraction*100:.1f}%  |  "
        f"Grid energy: {plan.grid_energy_kwh:.1f} kWh  |  "
        f"Surplus: {plan.total_surplus_kwh:.1f} kWh  |  "
        f"Shortage: {plan.total_shortage_kwh:.1f} kWh"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Community Profile
# ─────────────────────────────────────────────────────────────────────────────

def page_community_profile(ctx: dict) -> None:
    section_header(
        "🏘️ Community Profile",
        "Configure illustrative facilities, priorities and energy assumptions.",
    )
    synthetic_badge()
    info_banner(
        "All demand and solar figures are illustrative synthetic values. "
        "They do not represent measurements from any real site."
    )

    facilities = get_facilities()

    st.markdown("### Facility Overview")

    for i, fac in enumerate(facilities):
        with st.expander(f"{'🔴' if fac.priority=='critical' else '🟡' if fac.priority=='important' else '🟢'} {fac.label} ({fac.priority.capitalize()})", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_label = st.text_input("Name", value=fac.label, key=f"fac_label_{i}")
                new_priority = st.selectbox(
                    "Priority", ["critical", "important", "flexible"],
                    index=["critical", "important", "flexible"].index(fac.priority),
                    key=f"fac_priority_{i}",
                )
            with col2:
                new_min_e = st.number_input(
                    "Min Energy/day (kWh)", value=float(fac.min_energy_kwh_day),
                    min_value=0.0, max_value=500.0, step=1.0,
                    key=f"fac_mine_{i}",
                )
                new_peak = st.number_input(
                    "Peak Demand (kW)", value=float(fac.peak_demand_kw),
                    min_value=0.1, max_value=50.0, step=0.5,
                    key=f"fac_peak_{i}",
                )
            with col3:
                new_op_start = st.number_input(
                    "Operating Start (h)", value=fac.operating_hours[0],
                    min_value=0, max_value=23, step=1, key=f"fac_ops_{i}",
                )
                new_op_end = st.number_input(
                    "Operating End (h)", value=fac.operating_hours[1],
                    min_value=1, max_value=24, step=1, key=f"fac_ope_{i}",
                )
                new_flexible = st.checkbox(
                    "Flexible / Schedulable", value=fac.is_flexible, key=f"fac_flex_{i}",
                )

            if fac.key == "water_facility":
                st.caption("🚰 Water-specific parameters (illustrative assumptions)")
                wc1, wc2, wc3 = st.columns(3)
                with wc1:
                    wc_tank = st.number_input("Tank Capacity (L)", value=float(fac.tank_capacity_litres), min_value=0.0, step=1000.0, key=f"wc_tank_{i}")
                with wc2:
                    wc_daily = st.number_input("Daily Demand (L)", value=float(fac.daily_demand_litres), min_value=0.0, step=1000.0, key=f"wc_daily_{i}")
                with wc3:
                    wc_eff = st.number_input("Pump Efficiency (kWh/kL)", value=float(fac.pump_efficiency_kwh_per_kl), min_value=0.01, max_value=5.0, step=0.05, key=f"wc_eff_{i}")

                energy_for_water = (wc_daily / 1000) * wc_eff
                vol = estimate_water_volume(energy_for_water, wc_eff)
                st.caption(
                    f"ℹ Illustrative estimate: pumping {wc_daily:,.0f} L requires "
                    f"≈{energy_for_water:.1f} kWh (at {wc_eff} kWh/kL). "
                    "Real values require expert hydraulic assessment."
                )
            else:
                wc_tank, wc_daily, wc_eff = fac.tank_capacity_litres, fac.daily_demand_litres, fac.pump_efficiency_kwh_per_kl

            vres = validate_facility_config({
                "name": new_label, "priority": new_priority,
                "min_energy_kwh_day": new_min_e,
                "operating_hours": (new_op_start, new_op_end),
            })
            for err in vres.errors:
                danger_banner(err)
            for w in vres.warnings:
                warn_banner(w)

            if st.button(f"💾 Save changes", key=f"fac_save_{i}"):
                facilities[i] = Facility(
                    key=fac.key,
                    label=new_label,
                    priority=new_priority,
                    min_energy_kwh_day=new_min_e,
                    operating_hours=(int(new_op_start), int(new_op_end)),
                    peak_demand_kw=new_peak,
                    is_flexible=new_flexible,
                    scheduling_window=(int(new_op_start), int(new_op_end)) if new_flexible else None,
                    demand_col=fac.demand_col,
                    tank_capacity_litres=wc_tank,
                    daily_demand_litres=wc_daily,
                    pump_efficiency_kwh_per_kl=wc_eff,
                )
                st.session_state["facilities"] = facilities
                st.success(f"✅ {new_label} configuration saved.")

    st.markdown("---")
    st.markdown("### ➕ Add Custom Facility")
    with st.expander("Add a new facility"):
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            n_label = st.text_input("Facility Name", "Custom Facility", key="new_fac_label")
            n_priority = st.selectbox("Priority", ["critical", "important", "flexible"], index=2, key="new_fac_priority")
        with nc2:
            n_min_e = st.number_input("Min Energy/day (kWh)", 0.0, 500.0, 5.0, 1.0, key="new_fac_mine")
            n_peak = st.number_input("Peak Demand (kW)", 0.1, 50.0, 2.0, 0.5, key="new_fac_peak")
        with nc3:
            n_op_s = st.number_input("Op. Start (h)", 0, 23, 8, 1, key="new_fac_ops")
            n_op_e = st.number_input("Op. End (h)", 1, 24, 17, 1, key="new_fac_ope")
            n_flex = st.checkbox("Flexible", True, key="new_fac_flex")

        if st.button("➕ Add Facility", key="add_fac_btn"):
            key = n_label.lower().replace(" ", "_")[:20]
            new_fac = Facility(
                key=key, label=n_label, priority=n_priority,
                min_energy_kwh_day=n_min_e,
                operating_hours=(n_op_s, n_op_e),
                peak_demand_kw=n_peak, is_flexible=n_flex,
                scheduling_window=(n_op_s, n_op_e) if n_flex else None,
            )
            facilities.append(new_fac)
            st.session_state["facilities"] = facilities
            st.success(f"✅ Added {n_label}.")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Forecasts
# ─────────────────────────────────────────────────────────────────────────────

def page_forecasts(ctx: dict) -> None:
    section_header(
        "📈 Generation & Demand Forecasts",
        "7-day hourly solar generation and electricity demand projections.",
    )
    synthetic_badge()
    info_banner(
        "Forecasts are generated from SYNTHETIC historical data. "
        "Results are illustrative and must not be used for real operational decisions."
    )

    pipeline = get_pipeline(ctx["solar_scale"], ctx["demand_scale"])
    fc = pipeline.forecast_df
    raw = load_cached_data()

    tab1, tab2, tab3 = st.tabs(["📊 7-Day Forecast", "🔍 Baseline vs ML", "📏 Evaluation Metrics"])

    with tab1:
        st.markdown("### Hourly Solar Generation Forecast")
        fig_solar = go.Figure()
        fig_solar.add_trace(go.Scatter(
            x=fc.index, y=fc["forecast_solar_kw"] * ctx["solar_scale"],
            name="ML Forecast", line=dict(color="#1a7e5a", width=2),
        ))
        fig_solar.add_trace(go.Scatter(
            x=fc.index, y=fc["baseline_solar_kw"] * ctx["solar_scale"],
            name="Baseline (hour-mean)", line=dict(color="#2980b9", width=1.5, dash="dash"),
        ))
        # Uncertainty band
        fig_solar.add_trace(go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc["forecast_solar_high_kw"] * ctx["solar_scale"]) +
              list(fc["forecast_solar_low_kw"] * ctx["solar_scale"])[::-1],
            fill="toself", fillcolor="rgba(26,126,90,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1σ Uncertainty",
        ))
        fig_solar.update_layout(
            template="plotly_white", height=350,
            xaxis_title="Date", yaxis_title="Power (kW)",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_solar, use_container_width=True)

        st.markdown("### Hourly Demand Forecast")
        fig_demand = go.Figure()
        fig_demand.add_trace(go.Scatter(
            x=fc.index, y=fc["forecast_demand_kw"] * ctx["demand_scale"],
            name="ML Forecast", line=dict(color="#f0b429", width=2),
        ))
        fig_demand.add_trace(go.Scatter(
            x=fc.index, y=fc["baseline_demand_kw"] * ctx["demand_scale"],
            name="Baseline", line=dict(color="#e67e22", width=1.5, dash="dash"),
        ))
        fig_demand.add_trace(go.Scatter(
            x=list(fc.index) + list(fc.index[::-1]),
            y=list(fc["forecast_demand_high_kw"] * ctx["demand_scale"]) +
              list(fc["forecast_demand_low_kw"] * ctx["demand_scale"])[::-1],
            fill="toself", fillcolor="rgba(240,180,41,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±1σ Uncertainty",
        ))
        fig_demand.update_layout(
            template="plotly_white", height=350,
            xaxis_title="Date", yaxis_title="Power (kW)",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_demand, use_container_width=True)

        st.markdown("### Net Energy Balance")
        net = (fc["forecast_solar_kw"] * ctx["solar_scale"] -
               fc["forecast_demand_kw"] * ctx["demand_scale"])
        fig_net = px.area(
            x=fc.index, y=net,
            labels={"x": "Date", "y": "Net Balance (kW)"},
            color_discrete_sequence=["#1a7e5a"],
            template="plotly_white",
        )
        fig_net.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_net.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_net, use_container_width=True)

    with tab2:
        st.markdown("### Baseline vs ML Comparison (7-day forecast)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Solar Generation**")
            comp_solar = pd.DataFrame({
                "ML Forecast": fc["forecast_solar_kw"] * ctx["solar_scale"],
                "Baseline": fc["baseline_solar_kw"] * ctx["solar_scale"],
            })
            fig_c1 = px.line(comp_solar, template="plotly_white",
                             color_discrete_map={"ML Forecast": "#1a7e5a", "Baseline": "#2980b9"})
            fig_c1.update_layout(height=300, yaxis_title="kW",
                                  legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_c1, use_container_width=True)
        with col2:
            st.markdown("**Electricity Demand**")
            comp_demand = pd.DataFrame({
                "ML Forecast": fc["forecast_demand_kw"] * ctx["demand_scale"],
                "Baseline": fc["baseline_demand_kw"] * ctx["demand_scale"],
            })
            fig_c2 = px.line(comp_demand, template="plotly_white",
                             color_discrete_map={"ML Forecast": "#f0b429", "Baseline": "#e67e22"})
            fig_c2.update_layout(height=300, yaxis_title="kW",
                                  legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_c2, use_container_width=True)

        st.markdown("### Historical Actuals (last 7 days of training data)")
        last7 = raw[raw.index > raw.index.max() - pd.Timedelta(days=7)]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=last7.index, y=last7["solar_generation_kw"],
            name="Actual Solar", line=dict(color="#1a7e5a"),
        ))
        fig_hist.add_trace(go.Scatter(
            x=last7.index, y=last7["total_demand_kw"],
            name="Actual Demand", line=dict(color="#f0b429"),
        ))
        fig_hist.update_layout(template="plotly_white", height=280,
                                xaxis_title="Date", yaxis_title="kW",
                                legend=dict(orientation="h"))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.markdown("### Forecast Evaluation Metrics")
        info_banner(
            "Metrics are computed on a chronological 14-day held-out test set from "
            "SYNTHETIC data only. They do not represent performance on real measurements."
        )
        metrics_df = pipeline.get_metrics_df()
        if len(metrics_df) > 0:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.caption(
                "MAE = Mean Absolute Error (kW). RMSE = Root Mean Square Error (kW). "
                "Lower is better. Evaluated on synthetic test data."
            )
        else:
            warn_banner("Not enough test data to compute evaluation metrics.")

        # Anomaly detection
        st.markdown("### 🔍 Optional Anomaly Detection (Threshold-Based)")
        info_banner(
            "Module F: Simple statistical anomaly check on SYNTHETIC data only. "
            "Does NOT detect real equipment faults."
        )
        col = st.selectbox(
            "Select variable",
            ["solar_generation_kw", "total_demand_kw", "health_centre_demand_kw"],
        )
        try:
            anom_df = detect_anomalies(raw, col)
            n_anom = int(anom_df["anomaly"].sum())
            st.metric("Flagged anomalies (synthetic)", n_anom)
            if n_anom > 0:
                fig_a = go.Figure()
                fig_a.add_trace(go.Scatter(
                    x=anom_df.index, y=anom_df[col],
                    name=col, line=dict(color="#2980b9"),
                ))
                anom_only = anom_df[anom_df["anomaly"]]
                fig_a.add_trace(go.Scatter(
                    x=anom_only.index, y=anom_only[col],
                    mode="markers", name="Flagged",
                    marker=dict(color="#c0392b", size=8, symbol="x"),
                ))
                fig_a.update_layout(template="plotly_white", height=280)
                st.plotly_chart(fig_a, use_container_width=True)
        except Exception as e:
            warn_banner(f"Anomaly detection error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: Essential Services
# ─────────────────────────────────────────────────────────────────────────────

def page_essential_services(ctx: dict) -> None:
    section_header(
        "🔒 Essential Services Protection",
        "Critical, Important and Flexible load coverage against forecast supply.",
    )
    synthetic_badge()

    pipeline = get_pipeline(ctx["solar_scale"], ctx["demand_scale"])
    fc = pipeline.forecast_df
    facilities = get_facilities()

    service_report = evaluate_service_coverage(
        fc, facilities, ctx["solar_scale"], ctx["demand_scale"]
    )

    # Critical shortage banner
    if service_report.has_critical_shortage:
        danger_banner(
            f"🚨 {len(service_report.critical_shortages)} critical service shortage(s) detected. "
            "Review details below."
        )
    else:
        ok_banner("All critical service minimum requirements are projected to be met.")

    # Priority legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔴 **Critical** — Hard minimum; must be met.")
        crit_facs = [f for f in facilities if f.priority == "critical"]
        for f in crit_facs:
            st.caption(f"• {f.label}: ≥{f.min_energy_kwh_day:.0f} kWh/day")
    with col2:
        st.markdown("🟡 **Important** — Schedulable within safe windows.")
        imp_facs = [f for f in facilities if f.priority == "important"]
        for f in imp_facs:
            st.caption(f"• {f.label}: ≥{f.min_energy_kwh_day:.0f} kWh/day")
    with col3:
        st.markdown("🟢 **Flexible** — Deferrable loads.")
        flex_facs = [f for f in facilities if f.priority == "flexible"]
        for f in flex_facs:
            st.caption(f"• {f.label}")

    st.markdown("---")

    # Coverage table
    st.markdown("### Daily Service Coverage Table")
    cov_df = service_report.summary_df()
    if len(cov_df) > 0:
        def highlight_shortage(row):
            if row["Status"] == "⚠ SHORTAGE":
                return ["background-color: #fdecea"] * len(row)
            return [""] * len(row)
        st.dataframe(
            cov_df.style.apply(highlight_shortage, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        warn_banner("No coverage data available.")

    # Coverage chart
    st.markdown("### Coverage by Facility (7-Day Average)")
    if len(cov_df) > 0:
        avg_cov = cov_df.groupby("Facility")["Coverage (%)"].mean().reset_index()
        avg_cov.columns = ["Facility", "Avg Coverage (%)"]
        fig = px.bar(
            avg_cov, x="Facility", y="Avg Coverage (%)",
            color="Avg Coverage (%)",
            color_continuous_scale=["#c0392b", "#f0b429", "#1a7e5a"],
            range_color=[0, 100],
            template="plotly_white",
        )
        fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="100% target")
        fig.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Shortage details
    if service_report.has_critical_shortage:
        st.markdown("### ⚠️ Critical Shortage Details")
        for cs in service_report.critical_shortages:
            with st.expander(f"🚨 {cs.facility_label} — {cs.date.date()}", expanded=True):
                danger_banner(cs.warning_message)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Required", f"{cs.required_kwh:.1f} kWh")
                cc2.metric("Available", f"{cs.available_kwh:.1f} kWh")
                cc3.metric("Shortage", f"{cs.shortage_kwh:.1f} kWh", delta_color="inverse")
                st.caption(
                    "Priority: CRITICAL. This shortage must not be silently ignored. "
                    "Consider adding storage, grid supply or reducing other loads."
                )

    # Warnings list
    real_warnings = [w for w in service_report.warnings if w.strip()]
    if real_warnings:
        st.markdown("### All Warnings")
        for w in real_warnings:
            warn_banner(w)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: Weekly Plan
# ─────────────────────────────────────────────────────────────────────────────

def page_weekly_plan(ctx: dict) -> None:
    section_header(
        "📅 7-Day Energy Plan",
        "Rolling weekly schedule with hourly energy balance and recommendations.",
    )
    synthetic_badge()

    pipeline = get_pipeline(ctx["solar_scale"], ctx["demand_scale"])
    fc = pipeline.forecast_df
    facilities = get_facilities()

    if st.button("🔄 Regenerate Plan", type="primary"):
        st.cache_data.clear()
        st.rerun()

    plan = build_weekly_plan(
        fc, facilities,
        solar_scale=ctx["solar_scale"],
        demand_scale=ctx["demand_scale"],
        grid_capacity_kw=ctx["grid_capacity_kw"],
        storage_kwh=ctx["storage_kwh"],
        grid_available=ctx["grid_available"],
    )

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌿 Renewable Fraction", f"{plan.renewable_fraction*100:.1f}%")
    m2.metric("⚡ Total Surplus", f"{plan.total_surplus_kwh:.1f} kWh")
    m3.metric("⚠ Total Shortage", f"{plan.total_shortage_kwh:.1f} kWh",
              delta_color="inverse" if plan.total_shortage_kwh > 0 else "normal")
    m4.metric("🔌 Grid Energy", f"{plan.grid_energy_kwh:.1f} kWh")

    # Warnings
    real_warnings = [w for w in plan.shortage_warnings if w.strip()]
    if real_warnings:
        for w in real_warnings:
            warn_banner(w)

    st.markdown("---")

    # Daily summary
    st.markdown("### 📊 Daily Summary")
    daily_disp = plan.daily_summary_display()
    st.dataframe(daily_disp, use_container_width=True)

    # Hourly chart
    st.markdown("### ⏰ Hourly Energy Balance")
    h = plan.hourly_df
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=h.index, y=h["solar_kw"], name="Solar", fill="tozeroy",
        line=dict(color="#1a7e5a"), fillcolor="rgba(26,126,90,0.15)",
    ))
    fig_hourly.add_trace(go.Scatter(
        x=h.index, y=h["demand_kw"], name="Demand", line=dict(color="#f0b429", width=2),
    ))
    fig_hourly.add_trace(go.Scatter(
        x=h.index, y=h["grid_kw"], name="Grid Import", line=dict(color="#2980b9", dash="dot"),
    ))
    if ctx["storage_kwh"] > 0:
        fig_hourly.add_trace(go.Scatter(
            x=h.index, y=h["storage_soc_kwh"],
            name="Storage SoC (kWh)", yaxis="y2",
            line=dict(color="#8e44ad", dash="dash"),
        ))
    if plan.total_shortage_kwh > 0:
        fig_hourly.add_trace(go.Scatter(
            x=h.index, y=h["unmet_kw"], name="Unmet Demand",
            fill="tozeroy", fillcolor="rgba(192,57,43,0.20)",
            line=dict(color="#c0392b"),
        ))

    layout_kwargs = dict(
        template="plotly_white", height=420,
        xaxis_title="Date/Hour", yaxis_title="Power (kW)",
        legend=dict(orientation="h", y=1.05),
    )
    if ctx["storage_kwh"] > 0:
        layout_kwargs["yaxis2"] = dict(
            title="SoC (kWh)", overlaying="y", side="right", showgrid=False,
        )
    fig_hourly.update_layout(**layout_kwargs)
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Critical reserve
    st.markdown("### 🔴 Critical Service Reserves")
    info_banner(
        f"Critical daily energy requirement: {plan.critical_reserve_kwh_day:.1f} kWh/day. "
        "This must be met before any other loads are served."
    )

    # Flexible load recommendations
    if plan.flex_recommendations:
        st.markdown("### 💡 Flexible-Load Scheduling Recommendations")
        rec_df = pd.DataFrame(plan.flex_recommendations)
        rec_df.columns = [c.replace("_", " ").title() for c in rec_df.columns]
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
        st.caption(
            "Recommendations are based on forecast solar surplus windows. "
            "All values are illustrative."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: Scenario Lab
# ─────────────────────────────────────────────────────────────────────────────

def page_scenario_lab(ctx: dict) -> None:
    section_header(
        "🔬 Scenario Laboratory",
        "Compare scenarios and scheduling methods side by side.",
    )
    synthetic_badge()
    info_banner(
        "All scenario values are illustrative defaults. Edit them freely. "
        "Results change in response to inputs — no values are hardcoded."
    )

    pipeline = get_pipeline(ctx["solar_scale"], ctx["demand_scale"])
    fc = pipeline.forecast_df
    facilities = get_facilities()

    # Scenario picker
    st.markdown("### Select or Configure Scenario")
    scenario_names = [s.name for s in PRESET_SCENARIOS] + ["Custom…"]
    chosen = st.selectbox("Preset Scenario", scenario_names, key="scen_pick")

    if chosen == "Custom…":
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            c_solar = st.slider("Solar Scale", 0.1, 1.5, ctx["solar_scale"], 0.05, key="c_solar")
            c_demand = st.slider("Demand Scale", 0.5, 2.0, ctx["demand_scale"], 0.05, key="c_demand")
        with cc2:
            c_storage = st.number_input("Storage (kWh)", 0.0, 500.0, float(ctx["storage_kwh"]), 5.0, key="c_stor")
            c_grid_avail = st.toggle("Grid Available", ctx["grid_available"], key="c_grid")
        with cc3:
            c_grid_cap = st.slider("Grid Capacity (kW)", 0.0, 20.0, ctx["grid_capacity_kw"], 0.5, key="c_gcap", disabled=not c_grid_avail)

        vres = validate_scenario_params({
            "solar_scale": c_solar, "demand_scale": c_demand,
            "storage_kwh": c_storage, "grid_available": c_grid_avail,
        })
        for e in vres.errors:
            danger_banner(e)

        scenario = build_custom_scenario(
            name="Custom", solar_scale=c_solar, demand_scale=c_demand,
            storage_kwh=c_storage, grid_available=c_grid_avail,
            grid_capacity_kw=c_grid_cap if c_grid_avail else 0.0,
        )
    else:
        scenario = next(s for s in PRESET_SCENARIOS if s.name == chosen)
        st.info(f"**{scenario.description}**  |  Notes: {scenario.notes}")

    run_btn = st.button("▶ Run Scenario", type="primary", key="run_scen_btn")

    if run_btn or "scen_result" not in st.session_state:
        with st.spinner("Running scenario…"):
            result = run_scenario(scenario, fc, facilities)
            st.session_state["scen_result"] = result

    if "scen_result" not in st.session_state:
        return

    result = st.session_state["scen_result"]
    plan = result.plan
    svc = result.service_report

    st.markdown("---")
    st.markdown(f"### Results: {result.scenario.name}")

    # Key metrics comparison
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shortage", f"{plan.total_shortage_kwh:.1f} kWh",
              delta_color="inverse" if plan.total_shortage_kwh > 0 else "normal")
    m2.metric("Renewable Frac.", f"{plan.renewable_fraction*100:.1f}%")
    m3.metric("Grid Energy", f"{plan.grid_energy_kwh:.1f} kWh")
    m4.metric("Critical Shortages", len(svc.critical_shortages),
              delta_color="inverse" if svc.critical_shortages else "normal")

    if svc.has_critical_shortage:
        danger_banner(
            f"🚨 {len(svc.critical_shortages)} critical service shortage(s) in this scenario."
        )
    else:
        ok_banner("No critical shortages in this scenario.")

    # Scheduling method comparison
    st.markdown("### 📊 Scheduling Method Comparison")
    comp_df = result.comparison_df()
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Comparison chart
    fig_comp = px.bar(
        comp_df.melt(id_vars=["Method"], value_vars=[
            "Total Unmet Demand (kWh)", "Grid Energy Used (kWh)",
            "Solar Utilised (kWh)",
        ]),
        x="Method", y="value", color="variable", barmode="group",
        labels={"value": "kWh", "variable": "Metric"},
        template="plotly_white",
        color_discrete_sequence=["#c0392b", "#2980b9", "#1a7e5a"],
    )
    fig_comp.update_layout(height=380, legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Infeasibility warnings
    for r in [result.schedule_fixed, result.schedule_rule, result.schedule_opt]:
        for note in r.infeasibility_notes:
            warn_banner(note)

    # Scenario effect on essential services
    st.markdown("### 🔒 Scenario Effect on Essential Services")
    cov_df = svc.summary_df()
    if len(cov_df) > 0:
        def highlight_shortage(row):
            return ["background-color: #fdecea"] * len(row) if row["Status"] == "⚠ SHORTAGE" else [""] * len(row)
        st.dataframe(
            cov_df.style.apply(highlight_shortage, axis=1),
            use_container_width=True, hide_index=True,
        )

    # Daily plan chart
    st.markdown("### 📅 Daily Plan (Optimised)")
    opt_daily = result.schedule_opt.daily_df
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(x=opt_daily.index, y=opt_daily["solar_kwh"], name="Solar", marker_color="#1a7e5a"))
    fig_daily.add_trace(go.Bar(x=opt_daily.index, y=opt_daily["demand_kwh"], name="Demand", marker_color="#f0b429"))
    fig_daily.add_trace(go.Bar(x=opt_daily.index, y=opt_daily["unmet_kwh"], name="Unmet", marker_color="#c0392b"))
    fig_daily.update_layout(
        barmode="group", template="plotly_white", height=340,
        xaxis_title="Day", yaxis_title="Energy (kWh)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7: Reports
# ─────────────────────────────────────────────────────────────────────────────

def page_reports(ctx: dict) -> None:
    section_header(
        "📄 Reports & Exports",
        "Download data, forecasts and a PDF summary report.",
    )
    synthetic_badge()
    warn_banner(
        "All exported data are SYNTHETIC and illustrative. "
        "Do not use for engineering design, procurement or policy decisions."
    )

    pipeline = get_pipeline(ctx["solar_scale"], ctx["demand_scale"])
    fc = pipeline.forecast_df
    facilities = get_facilities()

    plan = build_weekly_plan(
        fc, facilities,
        solar_scale=ctx["solar_scale"],
        demand_scale=ctx["demand_scale"],
        grid_capacity_kw=ctx["grid_capacity_kw"],
        storage_kwh=ctx["storage_kwh"],
        grid_available=ctx["grid_available"],
    )
    service_report = evaluate_service_coverage(
        fc, facilities, ctx["solar_scale"], ctx["demand_scale"]
    )

    r1, r2, r3 = compare_schedules(
        fc, facilities,
        solar_scale=ctx["solar_scale"],
        demand_scale=ctx["demand_scale"],
        grid_capacity_kw=ctx["grid_capacity_kw"],
        storage_kwh=ctx["storage_kwh"],
    )
    comp_df = pd.DataFrame([r1.summary_dict(), r2.summary_dict(), r3.summary_dict()])
    balance = energy_balance_summary(fc, ctx["solar_scale"], ctx["demand_scale"])

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Forecast CSV")
        st.caption("7-day hourly solar and demand forecast.")
        csv_fc = export_forecast_csv(fc)
        st.download_button(
            "⬇ Download Forecast CSV",
            data=csv_fc,
            file_name="suryasetu_forecast.csv",
            mime="text/csv",
            key="dl_forecast",
        )

    with col2:
        st.markdown("### 📅 Weekly Plan CSV")
        st.caption("Hourly and daily plan with energy balance.")
        csv_plan = export_plan_csv(plan.hourly_df, plan.daily_df)
        st.download_button(
            "⬇ Download Plan CSV",
            data=csv_plan,
            file_name="suryasetu_weekly_plan.csv",
            mime="text/csv",
            key="dl_plan",
        )

    with col3:
        st.markdown("### 📋 Service Coverage CSV")
        st.caption("Essential-service daily coverage table.")
        cov_df = service_report.summary_df()
        if len(cov_df) > 0:
            csv_cov = cov_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Coverage CSV",
                data=csv_cov,
                file_name="suryasetu_service_coverage.csv",
                mime="text/csv",
                key="dl_cov",
            )
        else:
            st.caption("No coverage data.")

    st.markdown("---")
    st.markdown("### 📄 PDF Summary Report")
    if REPORTLAB_AVAILABLE:
        info_banner("ReportLab is available. Click below to generate a PDF summary.")
        suffix = ".pdf"
        mime = "application/pdf"
    else:
        warn_banner(
            "ReportLab not installed. Generating plain-text report instead. "
            "Install reportlab to enable PDF export."
        )
        suffix = ".txt"
        mime = "text/plain"

    if st.button("📄 Generate Report", key="gen_pdf"):
        with st.spinner("Generating report…"):
            scen_name = f"Solar ×{ctx['solar_scale']}, Demand ×{ctx['demand_scale']}"
            pdf_bytes = export_pdf(
                scenario_name=scen_name,
                scenario_notes=f"Storage: {ctx['storage_kwh']} kWh. Grid: {'Yes' if ctx['grid_available'] else 'No'}.",
                balance_metrics=balance,
                daily_summary_df=plan.daily_df.round(2),
                coverage_df=cov_df if len(cov_df) > 0 else None,
                comparison_df=comp_df,
                forecast_metrics_df=pipeline.get_metrics_df(),
            )
        st.download_button(
            f"⬇ Download Report ({suffix})",
            data=pdf_bytes,
            file_name=f"suryasetu_report{suffix}",
            mime=mime,
            key="dl_pdf",
        )

    st.markdown("---")
    st.markdown("### ℹ Methodology, Assumptions & Limitations")
    with st.expander("Read methodology"):
        st.markdown("""
**Forecasting**
- Baseline: same-hour-of-day historical mean from training data.
- ML: scikit-learn RandomForestRegressor, chronological 80/20 split, evaluated on held-out 14-day test set.
- Uncertainty bands: ±1σ from ensemble of decision-tree predictions.

**Essential Services**
- Critical loads carry hard minimum-energy constraints.
- If supply cannot meet critical minimums, the shortfall is reported explicitly — never silently hidden.
- Water volume estimates use an illustrative pump-efficiency assumption (kWh/kL). Real water-supply systems require expert hydraulic assessment.

**Optimisation (Module D)**
- PuLP LP solver minimises: unmet demand + 0.5 × grid consumption.
- Hard constraints for critical-facility operating hours.
- If scenario is infeasible, the conflict is exposed.

**Scenario Lab**
- Default values are illustrative, not measured facts.
- Results respond to input changes in real time.

**Limitations**
- All data is synthetic. No real measurements, pilot results, cost savings or government partnerships are claimed.
- This is a decision-support prototype. It does not control physical equipment.
- Real deployment requires expert electrical, hydraulic and safety engineering review.
- Electricity sharing between facilities requires appropriate infrastructure and regulatory approval.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    init_state()
    ctx = render_sidebar()

    # Validate data on first load
    try:
        raw = load_cached_data()
        vres = validate_dataframe(raw)
        if not vres.is_valid:
            for e in vres.errors:
                st.error(f"Data error: {e}")
            st.stop()
        for w in vres.warnings:
            if w:
                st.warning(w)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    page = ctx["page"]

    if page == "🏠 Overview":
        page_overview(ctx)
    elif page == "🏘️ Community Profile":
        page_community_profile(ctx)
    elif page == "📈 Forecasts":
        page_forecasts(ctx)
    elif page == "🔒 Essential Services":
        page_essential_services(ctx)
    elif page == "📅 Weekly Plan":
        page_weekly_plan(ctx)
    elif page == "🔬 Scenario Lab":
        page_scenario_lab(ctx)
    elif page == "📄 Reports":
        page_reports(ctx)


if __name__ == "__main__":
    main()
