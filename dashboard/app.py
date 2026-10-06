"""
Building 59 BEMS — AI-Based Energy Optimization Dashboard (Phase 5).
An interactive research simulation platform exposing the complete predictive and counterfactual
optimization pipeline for Lawrence Berkeley National Laboratory (LBNL) Building 59 South Wing.

RESEARCH SIMULATION DISCLAIMER:
All delta energy values and recommended setpoints are model-based counterfactual estimates
derived from an empirical Ridge surrogate under strict operational safety bounds.
These outputs represent research simulation scenarios, NOT experimentally verified physical control.
"""

import sys
from pathlib import Path

# Add workspace root to sys.path to ensure module imports resolve cleanly
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    ELECTRICITY_TARIFF,
    EMISSION_FACTOR,
    CONTROL_BOUNDS,
    RAMP_LIMITS,
    T_CLASS,
    T_SAFETY,
    MIN_IMPROVEMENT_EPSILON,
    BASELOAD_ENERGY_FLOOR
)
from dashboard.utils import (
    load_simulation_results,
    load_joint_test_data,
    get_cached_models,
    load_benchmark_metrics,
    load_robustness_metrics,
    load_phase4_summary,
    run_scenario_optimization,
    get_representative_timestamps
)

# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLES
# ==============================================================================

st.set_page_config(
    page_title="Building 59 BEMS | AI Energy Optimization",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Metric Card Styling */
    .metric-card {
        background: #1e2430;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        color: #f8fafc;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-green { color: #10b981; }
    .delta-amber { color: #f59e0b; }
    .delta-blue { color: #3b82f6; }
    .delta-red { color: #ef4444; }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.03em;
    }
    .badge-safe { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-protected { background-color: #78350f; color: #fcd34d; border: 1px solid #d97706; }
    .badge-warning { background-color: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }
    .badge-neutral { background-color: #1e293b; color: #cbd5e1; border: 1px solid #475569; }

    /* Research Disclaimer Banner */
    .research-banner {
        background-color: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #3b82f6;
        padding: 10px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-bottom: 16px;
    }

    /* Section Subtitle */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 10px;
        margin-bottom: 12px;
        border-bottom: 1px solid #334155;
        padding-bottom: 4px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR NAVIGATION & GLOBAL METADATA
# ==============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🏢 Building 59 BEMS")
        st.markdown(
            "**AI-Based Energy Optimization**  \n"
            "*LBNL Building 59 — South Wing (Floors 3 & 4)*"
        )
        
        st.markdown(
            """
            <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; margin-bottom: 15px;">
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">SIMULATION RUNTIME</div>
                <div style="font-size: 0.9rem; color: #38bdf8; font-weight: 600;">Research Simulation (Frozen ML)</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Horizon: t → t+1 (1-Hour Ahead)</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Models: RF (T=0.51) | Ridge (α=10)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("### Navigation")
        pages = [
            "1. 📊 Executive Overview",
            "2. 🎛️ Live Scenario Simulator",
            "3. ⚡ Energy Analytics",
            "4. 👥 Occupancy Analytics",
            "5. 🧠 ML Model Performance",
            "6. 🎯 Optimization Analysis",
            "7. 🛡️ Safety & Validity",
            "8. 📖 About & Methodology"
        ]
        selected_page = st.radio("Select View:", pages, label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### Baseline Tariffs & Safeguards")
        st.markdown(f"- **Tariff**: ${ELECTRICITY_TARIFF:.2f}/kWh")
        st.markdown(f"- **Grid Carbon**: {EMISSION_FACTOR:.3f} kg CO₂/kWh")
        st.markdown(f"- **Safety Gate (T_safety)**: {T_SAFETY:.2f}")
        st.markdown(f"- **Energy Display Floor**: {BASELOAD_ENERGY_FLOOR:.2f} kWh")
        
        st.markdown("---")
        st.caption(
            "Phase 5 Streamlit Research Platform  \n"
            "Academic BEMS Framework © 2026"
        )
        return selected_page


# ==============================================================================
# HELPER RENDERING FUNCTIONS
# ==============================================================================

def render_disclaimer():
    st.markdown(
        """
        <div class="research-banner">
            ⚠️ <strong>RESEARCH SIMULATION NOTICE:</strong> All energy reductions, delta values, and setpoints are 
            <strong>model-predicted counterfactual estimates</strong> derived from an empirical linear surrogate. 
            They are not experimentally verified physical building measurements. No physical control signals are dispatched.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================

def page_executive_overview(sim_df: pd.DataFrame):
    st.title("📊 Executive BEMS Overview")
    render_disclaimer()
    
    st.markdown("### Selected Test-Period Operating State")
    
    # Allow user to pick a timestamp from test set to view instantaneous status
    col_sel, col_stat = st.columns([2, 1])
    with col_sel:
        rep_options = get_representative_timestamps()
        selected_key = st.selectbox(
            "Select Benchmark Timestamp Scenario:",
            list(rep_options.keys()),
            index=1
        )
        ts_str = rep_options[selected_key]
        ts = pd.to_datetime(ts_str)
        if ts not in sim_df.index:
            ts = sim_df.index[0]
            
    with col_stat:
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        gate_status = sim_df.loc[ts, "control_safety_gate"]
        conf_flag = sim_df.loc[ts, "confidence_flag"]
        
        if gate_status == "PROTECTED_OCCUPIED":
            badge_html = '<span class="badge badge-protected">🛡️ PROTECTED_OCCUPIED</span>'
        else:
            badge_html = '<span class="badge badge-safe">⚡ CONTROL-SAFE (SETBACK ELIGIBLE)</span>'
        st.markdown(f"**Safety Gate Status:** {badge_html}", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Extract row metrics
    row = sim_df.loc[ts]
    prob_occ = float(row["occupancy_probability"])
    occ_state = int(row["occupancy_state"])
    outdoor_t = float(row["outdoor_temp_c"])
    base_fan = float(row["baseline_fan"])
    rec_fan = float(row["recommended_fan"])
    base_damper = float(row["baseline_damper"])
    rec_damper = float(row["recommended_damper"])
    base_energy = float(row["baseline_predicted_energy"])
    opt_energy = float(row["optimized_predicted_energy"])
    delta_energy = float(row["optimized_delta_energy"])
    save_pct = float(row["estimated_saving_percent"])
    cost_save = float(row["estimated_cost_saving"])
    co2_save = float(row["estimated_co2_reduction"])
    
    # 8 KPI Cards
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Forecasted Occupancy</div>
                <div class="metric-value">{"OCCUPIED" if occ_state == 1 else "VACANT"}</div>
                <div class="metric-delta delta-amber">Prob: {prob_occ*100:.1f}% (T_safe={T_SAFETY:.2f})</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Outdoor Environment</div>
                <div class="metric-value">{outdoor_t:.1f} °C</div>
                <div class="metric-delta {'delta-red' if outdoor_t < 10 else 'delta-blue'}">
                    {"Cold Regime (<10°C)" if outdoor_t < 10 else "Mild Regime (≥10°C)"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Supply Fan VFD Setpoint</div>
                <div class="metric-value">{rec_fan:.0f}% <span style="font-size:0.9rem; color:#94a3b8;">(was {base_fan:.0f}%)</span></div>
                <div class="metric-delta {'delta-green' if rec_fan < base_fan else 'delta-neutral'}">
                    Δ: {rec_fan - base_fan:+.0f}% (Ramp ≤ {RAMP_LIMITS['fan_speed_max_delta']}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols[3]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Damper Outdoor Air</div>
                <div class="metric-value">{rec_damper:.0f}% <span style="font-size:0.9rem; color:#94a3b8;">(was {base_damper:.0f}%)</span></div>
                <div class="metric-delta {'delta-green' if rec_damper < base_damper else 'delta-neutral'}">
                    Δ: {rec_damper - base_damper:+.0f}% (Ramp ≤ {RAMP_LIMITS['damper_pct_max_delta']}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    kpi_cols2 = st.columns(4)
    with kpi_cols2[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Baseline Predicted Energy</div>
                <div class="metric-value">{base_energy:.2f} kWh</div>
                <div class="metric-delta delta-neutral">Floor: {BASELOAD_ENERGY_FLOOR:.2f} kWh</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols2[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Counterfactual Energy</div>
                <div class="metric-value">{opt_energy:.2f} kWh</div>
                <div class="metric-delta delta-green">Simulated Setback</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols2[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Hourly Savings</div>
                <div class="metric-value">{-delta_energy:.2f} kWh</div>
                <div class="metric-delta delta-green">Saved: {save_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi_cols2[3]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Economic / Carbon</div>
                <div class="metric-value">${cost_save:.2f}/hr</div>
                <div class="metric-delta delta-green">Avoided: {co2_save:.2f} kg CO₂/hr</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Test Period Trajectory Preview (994 Evaluated Hours)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["actual_total_energy"],
        mode="lines", name="Historical Actual Energy (kWh)",
        line=dict(color="#94a3b8", width=1, dash="dot"), opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["baseline_predicted_energy"],
        mode="lines", name="Model Baseline Display (kWh)",
        line=dict(color="#f59e0b", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["optimized_predicted_energy"],
        mode="lines", name="Counterfactual Optimized (kWh)",
        line=dict(color="#10b981", width=1.8)
    ))
    fig.update_layout(
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Simulation Timestamp",
        yaxis_title="Hourly Energy Consumption (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.info(
            f"**Action Recommendation for {ts_str}:**  \n"
            f"`{row['action_reason']}`"
        )
    with col_t2:
        st.warning(
            f"**Surrogate Validity & Confidence:**  \n"
            f"`{conf_flag}`  \n"
            f"Constraint Status: `{row['constraint_status']}`"
        )


# ==============================================================================
# PAGE 2: LIVE / SCENARIO SIMULATOR
# ==============================================================================

def page_scenario_simulator(joint_df: pd.DataFrame, occ_art: Any, energy_art: Any):
    st.title("🎛️ Interactive Scenario Simulator")
    render_disclaimer()
    
    st.markdown(
        "Select a verified historical timestamp from the test set to load its real sensor vector, "
        "optionally adjust setpoint overrides to explore counterfactual response, and trigger optimization."
    )
    
    rep_options = get_representative_timestamps()
    col_mode, col_pick = st.columns([1, 2])
    with col_mode:
        sel_mode = st.radio(
            "Scenario Selection Mode:",
            ["Preset Representative Scenarios", "Custom Test Timestamp"]
        )
    with col_pick:
        if sel_mode == "Preset Representative Scenarios":
            preset_choice = st.selectbox("Representative Scenario:", list(rep_options.keys()))
            selected_ts = pd.to_datetime(rep_options[preset_choice])
        else:
            all_ts = joint_df.index.tolist()
            idx_choice = st.select_slider(
                "Select Timestamp from Test Period:",
                options=all_ts,
                value=all_ts[100],
                format_func=lambda x: str(x)
            )
            selected_ts = pd.to_datetime(idx_choice)

    # Load historical row
    hist_row = joint_df.loc[selected_ts]
    if isinstance(hist_row, pd.DataFrame):
        hist_row = hist_row.iloc[0]
        
    st.markdown("---")
    st.markdown(f"#### 1. Historical Observed Sensor Vector at `{selected_ts}`")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Indoor Temp", f"{hist_row.get('indoor_temp_mean', 22.0):.1f} °C")
    c2.metric("Outdoor Temp", f"{hist_row.get('outdoor_temp_c', 12.0):.1f} °C")
    c3.metric("Rel Humidity", f"{hist_row.get('relative_humidity', 50.0):.1f} %")
    c4.metric("Solar Rad", f"{hist_row.get('solar_radiation', 0.0):.0f} W/m²")
    c5.metric("Baseline Fan", f"{hist_row.get('rtu_south_fan_spd_mean_next_hour', 65.0):.0f} %")
    c6.metric("Baseline Damper", f"{hist_row.get('rtu_south_damper_pct_mean_next_hour', 30.0):.0f} %")
    
    with st.expander("🛠️ Optional Counterfactual Overrides (What-If Exploration)", expanded=False):
        st.caption("Adjust inputs to simulate synthetic perturbations while enforcing safety and ramp limits.")
        ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
        override_fan = ov_c1.slider(
            "Override Baseline Fan Speed (%)", 
            min_value=40.0, max_value=90.0, 
            value=float(hist_row.get('rtu_south_fan_spd_mean_next_hour', 65.0)),
            step=5.0
        )
        override_damper = ov_c2.slider(
            "Override Baseline Damper (%)", 
            min_value=10.0, max_value=90.0, 
            value=float(hist_row.get('rtu_south_damper_pct_mean_next_hour', 30.0)),
            step=5.0
        )
        override_tin = ov_c3.slider(
            "Override Indoor Temp (°C)", 
            min_value=16.0, max_value=28.0, 
            value=float(hist_row.get('indoor_temp_mean', 22.0)),
            step=0.5
        )
        override_tout = ov_c4.slider(
            "Override Outdoor Temp (°C)", 
            min_value=2.0, max_value=30.0, 
            value=float(hist_row.get('outdoor_temp_c', 12.0)),
            step=0.5
        )
        use_overrides = st.checkbox("Apply these overrides to optimization", value=False)
        
    overrides_dict = None
    if use_overrides:
        overrides_dict = {
            'rtu_south_fan_spd_mean_next_hour': override_fan,
            'rtu_south_damper_pct_mean_next_hour': override_damper,
            'indoor_temp_mean': override_tin,
            'outdoor_temp_c': override_tout
        }

    st.markdown("#### 2. Optimization Trigger")
    opt_btn = st.button("🚀 OPTIMIZE BUILDING CONTROLS", type="primary", use_container_width=True)
    
    # Run optimization (either on button click or default initial view)
    res = run_scenario_optimization(
        timestamp=selected_ts,
        joint_df=joint_df,
        occ_art=occ_art,
        energy_art=energy_art,
        overrides=overrides_dict,
        enforce_ramps=True,
        enforce_comfort=True,
        include_lighting=True
    )
    
    st.markdown("#### 3. Recommended Operating State & Impact Analysis")
    
    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Supply Fan VFD Setpoint</div>
                <div class="metric-value">{res['recommended_fan']:.0f}% <span style="font-size:0.85rem; color:#94a3b8;">(was {res['baseline_fan']:.0f}%)</span></div>
                <div class="metric-delta {'delta-green' if res['recommended_fan'] < res['baseline_fan'] else 'delta-neutral'}">
                    Δ: {res['recommended_fan'] - res['baseline_fan']:+.0f}% (Bounded ≤ 15%/hr)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with res_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Damper Outdoor Air Setpoint</div>
                <div class="metric-value">{res['recommended_damper']:.0f}% <span style="font-size:0.85rem; color:#94a3b8;">(was {res['baseline_damper']:.0f}%)</span></div>
                <div class="metric-delta {'delta-green' if res['recommended_damper'] < res['baseline_damper'] else 'delta-neutral'}">
                    Δ: {res['recommended_damper'] - res['baseline_damper']:+.0f}% (Bounded ≤ 30%/hr)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with res_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model-Predicted Energy</div>
                <div class="metric-value">{res['optimized_predicted_energy']:.2f} kWh <span style="font-size:0.85rem; color:#94a3b8;">(was {res['baseline_predicted_energy']:.2f})</span></div>
                <div class="metric-delta delta-green">
                    ΔE: {res['optimized_delta_energy']:+.2f} kWh ({res['estimated_saving_percent']:.1f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with res_c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Rate Impacts</div>
                <div class="metric-value">${res['estimated_cost_saving']:.2f}/hr</div>
                <div class="metric-delta delta-green">
                    Avoided: {res['estimated_co2_reduction']:.2f} kg CO₂/hr
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("#### 4. Diagnostic Attribution: Why Was This Action Recommended?")
    col_att1, col_att2 = st.columns([2, 1])
    with col_att1:
        st.success(f"**Optimizer Recommendation Engine Explanation:**  \n{res['action_reason']}")
        st.info(
            f"**Safety Gating:** `{res['control_safety_gate']}`  \n"
            f"Occupancy Probability: `{res['occupancy_probability']*100:.1f}%` "
            f"(Decision Threshold $T_{{class}}={T_CLASS}$, Safety Threshold $T_{{safety}}={T_SAFETY}$)  \n"
            f"**Surrogate Confidence:** `{res['confidence_flag']}`"
        )
    with col_att2:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Baseline",
            x=["Fan (%)", "Damper (%)", "Energy (kWh)"],
            y=[res['baseline_fan'], res['baseline_damper'], res['baseline_predicted_energy']],
            marker_color="#f59e0b"
        ))
        fig_bar.add_trace(go.Bar(
            name="Recommended",
            x=["Fan (%)", "Damper (%)", "Energy (kWh)"],
            y=[res['recommended_fan'], res['recommended_damper'], res['optimized_predicted_energy']],
            marker_color="#10b981"
        ))
        fig_bar.update_layout(
            template="plotly_dark",
            barmode="group",
            height=220,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=1.2, x=0.2)
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# PAGE 3: ENERGY ANALYTICS
# ==============================================================================

def page_energy_analytics(sim_df: pd.DataFrame):
    st.title("⚡ Energy Analytics & Counterfactual Comparison")
    render_disclaimer()
    
    st.markdown("### Interactive Filters")
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        date_range = st.date_input(
            "Filter Date Range:",
            value=[sim_df.index.min().date(), sim_df.index.max().date()],
            min_value=sim_df.index.min().date(),
            max_value=sim_df.index.max().date()
        )
    with f_c2:
        occ_filter = st.selectbox("Occupancy State Filter:", ["All", "Occupied Only", "Unoccupied Only"])
    with f_c3:
        temp_filter = st.selectbox(
            "Outdoor Temperature Regime:",
            ["All", "Cold (<10°C)", "Mild (10-16°C)", "Warm (>16°C)"]
        )
        
    # Apply filtering safely
    f_df = sim_df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        f_df = f_df.loc[(f_df.index.date >= start_d) & (f_df.index.date <= end_d)]
        
    if occ_filter == "Occupied Only":
        f_df = f_df.loc[f_df["actual_occupancy_state"] == 1]
    elif occ_filter == "Unoccupied Only":
        f_df = f_df.loc[f_df["actual_occupancy_state"] == 0]
        
    if temp_filter == "Cold (<10°C)":
        f_df = f_df.loc[f_df["outdoor_temp_c"] < 10.0]
    elif temp_filter == "Mild (10-16°C)":
        f_df = f_df.loc[(f_df["outdoor_temp_c"] >= 10.0) & (f_df["outdoor_temp_c"] <= 16.0)]
    elif temp_filter == "Warm (>16°C)":
        f_df = f_df.loc[f_df["outdoor_temp_c"] > 16.0]
        
    st.caption(f"Displaying {len(f_df)} filtered hours out of {len(sim_df)} total test hours.")
    
    # Chart 1: Energy over time (3 lines)
    st.markdown("#### 1. Energy Trajectory: Actual vs Baseline vs Counterfactual")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=f_df.index, y=f_df["actual_total_energy"],
        mode="lines", name="Historical Actual (Total)",
        line=dict(color="#64748b", width=1, dash="dot"), opacity=0.8
    ))
    fig1.add_trace(go.Scatter(
        x=f_df.index, y=f_df["baseline_predicted_energy"],
        mode="lines", name="Model Baseline Display Prediction",
        line=dict(color="#f59e0b", width=1.5)
    ))
    fig1.add_trace(go.Scatter(
        x=f_df.index, y=f_df["optimized_predicted_energy"],
        mode="lines", name="Counterfactual Optimized Prediction",
        line=dict(color="#10b981", width=1.8)
    ))
    fig1.update_layout(
        template="plotly_dark", height=380,
        xaxis_title="Timestamp", yaxis_title="Hourly Energy (kWh)",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2-column layout for Charts 2 & 3
    col_c2, col_c3 = st.columns(2)
    with col_c2:
        st.markdown("#### 2. Actual vs Baseline Model Prediction")
        fig2 = px.scatter(
            f_df, x="actual_total_energy", y="baseline_predicted_energy",
            color="outdoor_temp_c",
            color_continuous_scale="Viridis",
            labels={"actual_total_energy": "Actual Energy (kWh)", "baseline_predicted_energy": "Baseline Pred (kWh)", "outdoor_temp_c": "Outdoor T (°C)"},
            template="plotly_dark",
            height=320
        )
        fig2.add_shape(
            type="line", line=dict(dash="dash", color="#ef4444", width=1.5),
            x0=0, y0=0, x1=float(f_df["actual_total_energy"].max() or 30), y1=float(f_df["actual_total_energy"].max() or 30)
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_c3:
        st.markdown("#### 3. Actual vs Optimized Counterfactual")
        fig3 = px.scatter(
            f_df, x="actual_total_energy", y="optimized_predicted_energy",
            color="outdoor_temp_c",
            color_continuous_scale="Plasma",
            labels={"actual_total_energy": "Actual Energy (kWh)", "optimized_predicted_energy": "Optimized Pred (kWh)", "outdoor_temp_c": "Outdoor T (°C)"},
            template="plotly_dark",
            height=320
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    # Charts 4 & 5: Delta E time series and Cumulative savings
    col_c4, col_c5 = st.columns(2)
    with col_c4:
        st.markdown("#### 4. Estimated Hourly ΔE (Savings)")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=f_df.index, y=f_df["optimized_delta_energy"],
            name="ΔE = E_cand - E_base",
            marker_color="#10b981"
        ))
        fig4.update_layout(
            template="plotly_dark", height=320,
            xaxis_title="Timestamp", yaxis_title="Estimated ΔE (kWh)",
            yaxis=dict(zeroline=True, zerolinecolor="#ef4444")
        )
        st.plotly_chart(fig4, use_container_width=True)
        
    with col_c5:
        st.markdown("#### 5. Cumulative Estimated Energy Savings")
        cum_savings = (-f_df["optimized_delta_energy"]).cumsum()
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=f_df.index, y=cum_savings,
            mode="lines", name="Cumulative Savings (kWh)",
            line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.15)"
        ))
        fig5.update_layout(
            template="plotly_dark", height=320,
            xaxis_title="Timestamp", yaxis_title="Cumulative Saved kWh"
        )
        st.plotly_chart(fig5, use_container_width=True)
        
    # Charts 6 & 7: Distribution and Submeter breakdown
    col_c6, col_c7 = st.columns(2)
    with col_c6:
        st.markdown("#### 6. Hourly Savings Distribution")
        fig6 = px.histogram(
            f_df.loc[f_df["estimated_energy_saving"] > 0],
            x="estimated_energy_saving",
            nbins=35,
            template="plotly_dark",
            labels={"estimated_energy_saving": "Estimated Energy Saving (kWh)"},
            color_discrete_sequence=["#10b981"],
            height=320
        )
        fig6.update_layout(xaxis_title="Hourly Estimated Saving (kWh)", yaxis_title="Frequency (Hours)")
        st.plotly_chart(fig6, use_container_width=True)
        
    with col_c7:
        st.markdown("#### 7. South Wing Submeter Breakdown (Actual)")
        if "hvac_actual_kwh" in f_df.columns:
            sub_agg = pd.DataFrame({
                "Submeter": ["HVAC (RTU South)", "Lighting", "Plug Loads (MELs)"],
                "Total Energy (kWh)": [
                    f_df["hvac_actual_kwh"].sum(),
                    f_df["lig_actual_kwh"].sum(),
                    f_df["mels_actual_kwh"].sum()
                ]
            })
            fig7 = px.pie(
                sub_agg, values="Total Energy (kWh)", names="Submeter",
                color_discrete_sequence=["#38bdf8", "#f59e0b", "#ec4899"],
                template="plotly_dark",
                hole=0.4,
                height=320
            )
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("Submeter breakdown data not loaded.")


# ==============================================================================
# PAGE 4: OCCUPANCY ANALYTICS
# ==============================================================================

def page_occupancy_analytics(sim_df: pd.DataFrame):
    st.title("👥 Occupancy Analytics & Classifier Evaluation")
    render_disclaimer()
    
    st.markdown(
        """
        The South Wing occupancy forecaster ($t \\to t+1$) acts as the master safety barrier for setback controls.
        It evaluates whether the upcoming operating hour is eligible for deep ventilation setbacks or requires
        comfort-protected airflow.
        """
    )
    
    # Time Series of Probability and States
    st.markdown("#### 1. Occupancy Probability Trajectory & Dual Safety Thresholds")
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["occupancy_probability"],
        mode="lines", name="Predicted Occupancy Probability P(Occ=1)",
        line=dict(color="#38bdf8", width=1.5)
    ))
    fig_occ.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["actual_occupancy_state"],
        mode="lines", name="Actual Ground-Truth Occupancy",
        line=dict(color="#94a3b8", width=1, dash="dot"), opacity=0.6
    ))
    # Add horizontal threshold lines
    fig_occ.add_hline(
        y=T_CLASS, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"Optimal Decision Threshold (T_class = {T_CLASS:.2f})",
        annotation_position="top left"
    )
    fig_occ.add_hline(
        y=T_SAFETY, line_dash="dash", line_color="#ef4444",
        annotation_text=f"Conservative Safety Threshold (T_safety = {T_SAFETY:.2f})",
        annotation_position="bottom left"
    )
    fig_occ.update_layout(
        template="plotly_dark", height=380,
        xaxis_title="Timestamp", yaxis_title="Probability [0.0, 1.0]",
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig_occ, use_container_width=True)
    
    # 2-column layout for Diurnal Profile and Weekly Heatmap
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### 2. Diurnal Hour-of-Day Occupancy Profile")
        df_profile = sim_df.copy()
        df_profile["hour"] = df_profile.index.hour
        df_profile["is_weekend"] = df_profile.index.dayofweek >= 5
        diurnal = df_profile.groupby(["hour", "is_weekend"])["actual_occupancy_state"].mean().reset_index()
        diurnal["Day Type"] = diurnal["is_weekend"].map({True: "Weekend", False: "Weekday"})
        
        fig_diurnal = px.line(
            diurnal, x="hour", y="actual_occupancy_state", color="Day Type",
            template="plotly_dark", markers=True,
            labels={"hour": "Hour of Day", "actual_occupancy_state": "Mean Occupancy Rate"},
            color_discrete_map={"Weekday": "#38bdf8", "Weekend": "#f59e0b"},
            height=320
        )
        st.plotly_chart(fig_diurnal, use_container_width=True)
        
    with col_d2:
        st.markdown("#### 3. Weekly Occupancy Pattern Heatmap")
        df_heat = sim_df.copy()
        df_heat["day_name"] = df_heat.index.day_name()
        df_heat["hour"] = df_heat.index.hour
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat_matrix = df_heat.groupby(["day_name", "hour"])["occupancy_probability"].mean().unstack()
        heat_matrix = heat_matrix.reindex(day_order)
        
        fig_heat = px.imshow(
            heat_matrix,
            labels=dict(x="Hour of Day", y="Day of Week", color="Mean P(Occ)"),
            template="plotly_dark",
            color_continuous_scale="Viridis",
            height=320
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 4. Champion Random Forest Classifier Frozen Test Metrics")
    
    # Load Phase 3 benchmark results
    p3_res = load_benchmark_metrics()
    occ_test = p3_res.get("occupancy_classification", {}).get("test_results", {})
    
    m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
    m_c1.metric("F1 Score (T=0.51)", f"{occ_test.get('Test F1 (Tuned)', 0.9419):.4f}")
    m_c2.metric("Precision", f"{occ_test.get('Test Precision', 0.9332):.4f}")
    m_c3.metric("Recall", f"{occ_test.get('Test Recall', 0.9508):.4f}")
    m_c4.metric("ROC-AUC", f"{occ_test.get('ROC-AUC', 0.9392):.4f}")
    m_c5.metric("PR-AUC", f"{occ_test.get('PR-AUC', 0.9806):.4f}")
    m_c6.metric("Balanced Acc", f"{occ_test.get('Bal Acc', 0.8643):.4f}")
    
    st.markdown(
        """
        <div style="background-color: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 18px; border-radius: 4px; margin-top: 15px;">
            <strong>DUAL-THRESHOLD ASYMMETRIC SAFETY ARCHITECTURE:</strong><br>
            <ul>
                <li><strong>Classification Decision Threshold (T_class = 0.51):</strong> Derived via validation tuning to maximize balanced F1-score (0.9419 on test). Used for reporting and baseline feature vector preparation.</li>
                <li><strong>Safety Gating Threshold (T_safety = 0.30):</strong> A deliberately conservative operational guardrail. When predicted probability exceeds 0.30, the optimizer flags the space as <code>PROTECTED_OCCUPIED</code>, prohibiting deep ventilation setbacks. This safety margin reduced false-setback occupied risk hours by <strong>72.55%</strong> during testing.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 5: ML MODEL PERFORMANCE
# ==============================================================================

def page_model_performance():
    st.title("🧠 ML Model Benchmark & Audit Registry")
    render_disclaimer()
    
    st.markdown(
        "Direct registry of frozen Phase 3 candidate benchmarks and Phase 3.5 forecast robustness audits. "
        "No models are retrained in this application."
    )
    
    p3 = load_benchmark_metrics()
    p3_rob = load_robustness_metrics()
    
    tab_occ, tab_eng, tab_base = st.tabs([
        "Occupancy Model Comparison",
        "Energy Surrogate Comparison",
        "Naive Baselines Benchmark"
    ])
    
    with tab_occ:
        st.markdown("#### Candidate Occupancy Model Benchmark (Validation Set)")
        val_occ_list = p3.get("occupancy_classification", {}).get("validation_results", [])
        if val_occ_list:
            df_val_occ = pd.DataFrame(val_occ_list)
            st.dataframe(df_val_occ, use_container_width=True)
            
            fig_occ_comp = px.bar(
                df_val_occ, x="Model", y="Val F1 (Tuned)",
                color="PR-AUC", template="plotly_dark",
                title="Validation F1-Score across Architectures",
                color_continuous_scale="Viridis",
                height=320
            )
            st.plotly_chart(fig_occ_comp, use_container_width=True)
            
        st.markdown("#### Frozen Test Set Verification (Random Forest Champion)")
        test_occ = p3.get("occupancy_classification", {}).get("test_results", {})
        st.json(test_occ)
        
    with tab_eng:
        st.markdown("#### Candidate Energy Model Benchmark (Validation Set)")
        val_eng_list = p3.get("energy_forecasting", {}).get("validation_results", [])
        if val_eng_list:
            df_val_eng = pd.DataFrame(val_eng_list)
            st.dataframe(df_val_eng, use_container_width=True)
            
            y_col = "Chained RMSE" if "Chained RMSE" in df_val_eng.columns else df_val_eng.columns[2]
            fig_eng_comp = px.bar(
                df_val_eng, x="Model", y=y_col,
                template="plotly_dark",
                title="Validation RMSE across Energy Model Candidates (Chained Pipeline)",
                color_discrete_sequence=["#f59e0b"],
                height=320
            )
            st.plotly_chart(fig_eng_comp, use_container_width=True)
            
        st.markdown("#### Frozen Test Set Results (Ridge Regression Champion - Chained Pipeline)")
        test_eng = p3.get("energy_forecasting", {}).get("final_test_metrics_chained", {})
        st.json(test_eng)
        
        st.info(
            "**Architecture Selection Rationale:**  \n"
            "Although tree ensembles achieved low validation errors, they exhibited non-physical step-function "
            "artifacts and wild extrapolation under counterfactual setpoints. The L2-regularized **Ridge Regression (α=10)** "
            "surrogate was selected as the champion optimizer model because its linear gradients maintain monotonicity "
            "and predictable physical sensitivity within the calibrated operational envelope."
        )

    with tab_base:
        st.markdown("#### Naive Forecasting Baselines Audit (Phase 3.5)")
        st.markdown(
            "To prevent overconfidence, candidate ML models were audited against simple statistical baselines:"
        )
        baselines = p3_rob.get("baselines_summary", {})
        for k, v in baselines.items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")
            
        st.markdown("#### Outdoor Temperature Regime Stratification (Test Period)")
        regimes = p3_rob.get("regime_analysis", {}).get("outdoor_temperature", [])
        if regimes:
            st.dataframe(pd.DataFrame(regimes), use_container_width=True)


# ==============================================================================
# PAGE 6: OPTIMIZATION ANALYSIS
# ==============================================================================

def page_optimization_analysis(sim_df: pd.DataFrame):
    st.title("🎯 Counterfactual Optimization Simulation Analysis")
    render_disclaimer()
    
    p4 = load_phase4_summary()
    kpis = p4.get("overall_simulation_metrics", {})
    
    st.markdown("### Test-Period Simulation Aggregate KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated Hours", f"{kpis.get('hours_evaluated', 994)} hrs")
    c2.metric("Hours Optimized", f"{kpis.get('hours_optimized', 804)} hrs", f"{kpis.get('pct_hours_improved', 80.89):.1f}% active")
    c3.metric("Baseline Preserved", f"{kpis.get('hours_no_change', 190)} hrs", "NO_CHANGE gate")
    c4.metric("Estimated Energy Savings", f"{kpis.get('total_estimated_energy_savings_kwh', 5120.45):,.1f} kWh", f"{kpis.get('overall_estimated_energy_savings_pct', 53.32):.1f}% reduction")
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mean Delta / Opt Hour", f"{kpis.get('mean_delta_per_optimized_hour_kwh', -6.37):.2f} kWh")
    c6.metric("Median Delta / Opt Hour", f"{kpis.get('median_delta_per_optimized_hour_kwh', -6.58):.2f} kWh")
    c7.metric("Estimated Cost Savings", f"${kpis.get('total_estimated_cost_savings_usd', 1126.50):,.2f}")
    c8.metric("Estimated CO₂ Avoided", f"{kpis.get('total_estimated_co2_reduction_kg', 1075.29):,.1f} kg")
    
    st.markdown("---")
    st.markdown("### Regime Stratification Breakdown")
    col_reg1, col_reg2 = st.columns(2)
    
    reg_data = p4.get("regime_breakdown", {})
    with col_reg1:
        st.markdown("#### By Occupancy Regime")
        occ_breakdown = [
            reg_data.get("occupancy_occupied", {}),
            reg_data.get("occupancy_unoccupied", {})
        ]
        df_occ_b = pd.DataFrame(occ_breakdown)
        if not df_occ_b.empty:
            fig_occ_b = px.bar(
                df_occ_b, x="regime", y="savings_kwh",
                color="regime", template="plotly_dark",
                labels={"regime": "Occupancy Regime", "savings_kwh": "Estimated Savings (kWh)"},
                color_discrete_map={"Occupied": "#38bdf8", "Unoccupied": "#10b981"},
                height=280
            )
            st.plotly_chart(fig_occ_b, use_container_width=True)
            
    with col_reg2:
        st.markdown("#### By Temperature Regime")
        temp_breakdown = [
            reg_data.get("temp_cold", {}),
            reg_data.get("temp_mild", {}),
            reg_data.get("temp_warm", {})
        ]
        df_temp_b = pd.DataFrame(temp_breakdown)
        if not df_temp_b.empty:
            fig_temp_b = px.bar(
                df_temp_b, x="regime", y="savings_kwh",
                color="regime", template="plotly_dark",
                labels={"regime": "Temperature Regime", "savings_kwh": "Estimated Savings (kWh)"},
                color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
                height=280
            )
            st.plotly_chart(fig_temp_b, use_container_width=True)
            
    st.markdown("---")
    st.markdown("### Safeguard Ablation Study: Proving Physical Feasibility")
    
    ablations = p4.get("ablations", {})
    ablation_rows = []
    for k, v in ablations.items():
        ablation_rows.append({
            "Ablation Scenario": k,
            "Hours Optimized": v.get("hours_optimized"),
            "Estimated Savings (kWh)": v.get("total_estimated_energy_savings_kwh"),
            "Estimated Saving %": v.get("overall_estimated_energy_savings_pct"),
            "Cost Savings ($)": v.get("total_estimated_cost_savings_usd")
        })
    df_abl = pd.DataFrame(ablation_rows)
    st.dataframe(df_abl, use_container_width=True)
    
    st.markdown(
        """
        <div style="background-color: #450a0a; border-left: 4px solid #ef4444; padding: 12px 18px; border-radius: 4px; margin-top: 15px;">
            🚨 <strong>CRITICAL OBSERVABILITY CALLOUT: THE NON-PHYSICAL NO-RAMP ABLATION</strong><br>
            Notice <code>Ablation C (No Ramp Protection)</code> reports <strong>115.86% estimated energy savings (11,125.39 kWh)</strong>.
            This mathematically implausible figure occurs because unconstrained optimizers command instant jumps from 90% fan speed to 40% every single hour. 
            Real supply fans cannot violently step down by 50% in a single minute without tripping static pressure safety switches and causing severe duct buffeting.<br><br>
            <strong>The Full Safeguards Policy (Champion) restricts fan ramp to ≤ 15%/hr</strong>, producing the realistic, mechanically achievable 
            <strong>53.32% model-predicted reduction</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 7: SAFETY & VALIDITY
# ==============================================================================

def page_safety_validity():
    st.title("🛡️ Safety Constraints & Domain Validity")
    render_disclaimer()
    
    st.markdown(
        """
        Because machine learning models cannot understand fluid mechanics or building acoustics intrinsically,
        the optimizer is wrapped in a multi-layered deterministic safety envelope.
        """
    )
    
    st.markdown("### Operational Safety Bounds Envelope")
    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    b_c1.metric("Fan Speed Bounds", "40.0% – 90.0%", "Historical Support")
    b_c2.metric("Damper Bounds", "10.0% – 90.0%", "Ventilation Floor")
    b_c3.metric("Max Fan Ramp", "≤ 15.0% / hr", "Equipment Protection")
    b_c4.metric("Max Damper Ramp", "≤ 30.0% / hr", "Actuator Protection")
    
    b_c5, b_c6, b_c7, b_c8 = st.columns(4)
    b_c5.metric("Safety Threshold T_safety", f"{T_SAFETY:.2f}", "Occupancy Setback Guard")
    b_c6.metric("Improvement Epsilon", f"{MIN_IMPROVEMENT_EPSILON:.2f} kWh", "Prevents Hunting")
    b_c7.metric("Energy Display Floor", f"{BASELOAD_ENERGY_FLOOR:.2f} kWh", "Baseload Preservation")
    b_c8.metric("Grid Candidate Search", "99 Pairs", "Discrete Feasible Grid")
    
    st.markdown("---")
    st.markdown("### The Four System Operational States")
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown(
            """
            <div style="background-color: #064e3b; border: 1px solid #059669; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                <h4 style="color: #34d399; margin: 0 0 6px 0;">1. CONTROL-SAFE (SETBACK ELIGIBLE)</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0;">
                    Predicted occupancy probability is strictly below <code>T_safety = 0.30</code>. The zone is confirmed vacant. 
                    The optimizer is authorized to evaluate minimal fan speeds (down to 40%) and reduced ventilation dampers.
                </p>
            </div>
            <div style="background-color: #78350f; border: 1px solid #d97706; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                <h4 style="color: #fcd34d; margin: 0 0 6px 0;">2. PROTECTED_OCCUPIED</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0;">
                    Predicted occupancy probability is ≥ 0.30. Regardless of whether the space is heavily or lightly occupied, 
                    deep setback is prohibited. Candidate fan speeds must maintain adequate ventilation and thermal comfort standards.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with s_col2:
        st.markdown(
            """
            <div style="background-color: #7f1d1d; border: 1px solid #dc2626; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                <h4 style="color: #fca5a5; margin: 0 0 6px 0;">3. LOW_SURROGATE_CONFIDENCE (Cold Regime <10°C)</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0;">
                    Outdoor air temperature is below 10°C. In the training set (May–Nov 2018), cold weather was virtually absent. 
                    The linear surrogate exhibits negative bias in this regime. These predictions are tagged with low-confidence telemetry.
                </p>
            </div>
            <div style="background-color: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
                <h4 style="color: #cbd5e1; margin: 0 0 6px 0;">4. NO_CHANGE (Deadband Preservation)</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0;">
                    If no candidate setpoint yields an estimated improvement exceeding <code>ε = 0.20 kWh</code>, 
                    the optimizer commands <code>NO_CHANGE</code> to preserve existing equipment states and prevent valve hunting.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Domain Limitations & Scope Boundaries")
    st.warning(
        """
        1. **Empirical Surrogate vs Physics Simulator:** The Ridge energy model is an empirical regression surface, not an EnergyPlus or physical RC network model. It models energy consumption, but does not co-simulate zone thermal capacitance or heating reheat loops.
        2. **Indoor Temperature Dynamics:** Adjusting the supply fan setpoint does not update future indoor air temperatures within this simulation horizon.
        3. **No Closed-Loop Building Actuation:** This dashboard operates as an offline academic research simulation. No BACnet, Modbus, or MQTT signals are dispatched to physical building hardware.
        """
    )


# ==============================================================================
# PAGE 8: ABOUT / METHODOLOGY
# ==============================================================================

def page_about_methodology():
    st.title("📖 About & Modeling Methodology")
    render_disclaimer()
    
    st.markdown(
        """
        ### System Architecture & Pipeline Formulation
        This interactive platform encapsulates the complete end-to-end BEMS research workflow developed for 
        **LBNL Building 59 (South Wing, Floors 3 & 4)**.
        """
    )
    
    st.markdown(
        """
        ```
             Current Building State (t)
                    ↓
             Occupancy Forecast (t → t+1) [Random Forest, T_class=0.51, T_safety=0.30]
                    ↓
             Energy Forecast (t → t+1) [Ridge Regression, α=10]
                    ↓
             Counterfactual Optimization (Bounded Grid Search over 99 Candidates)
                    ↓
             Recommended Operating Setpoints (Fan Speed %, Damper %)
                    ↓
             Estimated Differential Savings (ΔE = E_cand - E_base)
        ```
        """
    )
    
    st.markdown("### Key Technical Specifications")
    st.markdown(
        """
        | Dimension | Specification | Notes |
        | :--- | :--- | :--- |
        | **Dataset** | LBNL Building 59 | Open-source multi-sensor commercial building dataset |
        | **Spatial Boundary** | South Wing (Floors 3 & 4) | High sensor density office and open workspace |
        | **Time Resolution** | 1 Hour (Uniform Resampling) | Leakage-safe backward-looking features |
        | **Prediction Horizon** | $t \\to t+1$ | Features up to $t$ forecast conditions at $t+1$ |
        | **Occupancy Champion** | Random Forest Classifier | Tuned threshold $T_{class}=0.51$, Safety $T_{safety}=0.30$ |
        | **Energy Champion** | Ridge Linear Surrogate | $\\alpha=10$, Robust monotonic gradient |
        | **Candidate Space** | 99 Combinations | Fan $\\in \\{40, 45, \\dots, 90\\}$, Damper $\\in \\{10, 20, \\dots, 90\\}$ |
        | **Electricity Tariff** | $0.22 / kWh | California Commercial / PG&E Commercial B-19 Schedule |
        | **Carbon Intensity** | 0.210 kg CO₂ / kWh | EPA eGRID CAMX Subregion Factor |
        | **Display Energy Floor** | 6.50 kWh | Calibrated baseload display floor |
        """
    )
    
    st.markdown("### Mathematical Formulation")
    st.latex(r"\Delta E(u) = \hat{E}(u_{\text{candidate}}, \hat{O}_{t+1}, X_t) - \hat{E}(u_{\text{baseline}}, \hat{O}_{t+1}, X_t)")
    st.markdown(
        """
        Where:
        - $u = (\\text{fan\\_spd}, \\text{damper\\_pct})$ is the controllable HVAC actuator vector.
        - $\\hat{O}_{t+1} \\in \\{0, 1\\}$ is the predicted occupancy state.
        - $X_t$ is the frozen exogenous context (outdoor temperature, solar radiation, humidity, calendar).
        - A negative $\\Delta E$ indicates estimated energy reduction.
        """
    )
    
    st.markdown("### Citation & Attribution")
    st.markdown(
        """
        If utilizing this methodology or simulation results in research publications:
        - **Dataset Source:** *Building 59 Multi-Sensor Dataset*, Lawrence Berkeley National Laboratory (LBNL).
        - **Pipeline Implementation:** Academic BEMS Energy Optimization Pipeline (Phases 0–5).
        """
    )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main():
    selected_page = render_sidebar()
    
    # Load required data and models safely with clear user feedback
    try:
        sim_df = load_simulation_results()
        joint_df = load_joint_test_data()
        occ_art, energy_art = get_cached_models()
    except Exception as e:
        st.error(f"❌ Failed to initialize dashboard resources: {str(e)}")
        st.info("Ensure all preprocessing and training phases have completed and models are serialized.")
        return

    if "1. 📊 Executive Overview" in selected_page:
        page_executive_overview(sim_df)
    elif "2. 🎛️ Live Scenario Simulator" in selected_page:
        page_scenario_simulator(joint_df, occ_art, energy_art)
    elif "3. ⚡ Energy Analytics" in selected_page:
        page_energy_analytics(sim_df)
    elif "4. 👥 Occupancy Analytics" in selected_page:
        page_occupancy_analytics(sim_df)
    elif "5. 🧠 ML Model Performance" in selected_page:
        page_model_performance()
    elif "6. 🎯 Optimization Analysis" in selected_page:
        page_optimization_analysis(sim_df)
    elif "7. 🛡️ Safety & Validity" in selected_page:
        page_safety_validity()
    elif "8. 📖 About & Methodology" in selected_page:
        page_about_methodology()


if __name__ == "__main__":
    main()
