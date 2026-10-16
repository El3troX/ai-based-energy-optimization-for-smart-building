"""
Building 59 BEMS -- AI-Based Energy Optimization Dashboard (Phase 5).
RESEARCH SIMULATION DISCLAIMER: All delta energy values and recommended setpoints are
model-based counterfactual estimates derived from an empirical Ridge surrogate.
These outputs represent research simulation scenarios, NOT experimentally verified physical control.
"""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any

from src.config import (
    ELECTRICITY_TARIFF,
    EMISSION_FACTOR,
    CONTROL_BOUNDS,
    RAMP_LIMITS,
    T_CLASS,
    T_SAFETY,
    MIN_IMPROVEMENT_EPSILON,
    BASELOAD_ENERGY_FLOOR,
)
from dashboard.utils import (
    load_simulation_results,
    load_joint_test_data,
    get_cached_models,
    load_benchmark_metrics,
    load_robustness_metrics,
    load_phase4_summary,
    run_scenario_optimization,
    get_representative_timestamps,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Building 59 BEMS | AI Energy Optimization",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# PLOTLY THEME
# =============================================================================

_TMPL = "plotly_dark"
_LAYOUT = dict(
    template=_TMPL,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#CBD5E1"),
    margin=dict(l=40, r=20, t=36, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(gridcolor="#1F2E44", zeroline=False),
    yaxis=dict(gridcolor="#1F2E44", zeroline=False),
)

C = dict(
    actual="#94A3B8",
    baseline="#F59E0B",
    optimized="#10B981",
    savings="#38BDF8",
    danger="#EF4444",
    occupied="#818CF8",
    vacant="#10B981",
)


def _fig(**kw) -> go.Figure:
    fig = go.Figure(**kw)
    fig.update_layout(**_LAYOUT)
    return fig


def _layout_extras() -> dict:
    """Extra layout overrides for px figures (exclude 'template' key)."""
    return {k: v for k, v in _LAYOUT.items() if k != "template"}


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='padding:4px 0 16px 0;'>"
            "<span style='font-size:1.35rem;font-weight:700;color:#F1F5F9;'>Building 59 BEMS</span><br>"
            "<span style='font-size:0.78rem;color:#64748B;font-weight:500;letter-spacing:0.04em;'>"
            "AI ENERGY OPTIMIZATION &middot; LBNL</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;'>"
            "<span style='background:#0F2E1A;color:#10B981;border:1px solid #065F46;"
            "border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;'>&#9679; LIVE SIM</span>"
            "<span style='background:#0F1E34;color:#38BDF8;border:1px solid #075985;"
            "border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;'>t&#8594;t+1 HORIZON</span>"
            "<span style='background:#1E1427;color:#818CF8;border:1px solid #4338CA;"
            "border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:600;'>FROZEN ML</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        pages = [
            "Executive Overview",
            "Scenario Simulator",
            "Energy Analytics",
            "Occupancy Analytics",
            "ML Model Performance",
            "Optimization Analysis",
            "Safety & Validity",
            "About & Methodology",
        ]
        icons = [
            ":material/dashboard:",
            ":material/tune:",
            ":material/bolt:",
            ":material/group:",
            ":material/psychology:",
            ":material/track_changes:",
            ":material/shield:",
            ":material/menu_book:",
        ]
        selected = st.radio(
            "nav",
            pages,
            format_func=lambda p: f"{icons[pages.index(p)]}  {p}",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**System parameters**")
        st.caption(
            f"Tariff: **${ELECTRICITY_TARIFF:.2f}/kWh** &middot; "
            f"Carbon: **{EMISSION_FACTOR:.3f} kg CO\u2082/kWh**"
        )
        st.caption(
            f"T\u209b\u2090\u2066\u2091\u2068\u2064\u2065: **{T_SAFETY:.2f}** &middot; "
            f"T\u209c\u2097\u2090\u209b\u209b: **{T_CLASS:.2f}** &middot; "
            f"\u03b5: **{MIN_IMPROVEMENT_EPSILON:.2f} kWh**"
        )
        st.caption(
            f"Energy floor: **{BASELOAD_ENERGY_FLOOR:.2f} kWh** &middot; "
            f"Fan ramp: **\u2264{RAMP_LIMITS['fan_speed_max_delta']}%/hr**"
        )
        st.divider()
        st.caption("Phase 5 Research Platform &middot; BEMS Framework \u00a9 2026")

    return selected


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _notice():
    st.info(
        ":material/science: **Research simulation notice:** All \u0394-energy values and setpoints "
        "are **model-predicted counterfactual estimates** from an empirical Ridge surrogate. "
        "No physical control signals are dispatched."
    )


def _section(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


# =============================================================================
# PAGE 1 - EXECUTIVE OVERVIEW
# =============================================================================

def page_executive_overview(sim_df: pd.DataFrame):
    st.title(":material/dashboard: Executive overview")
    _notice()

    rep_options = get_representative_timestamps()
    c_sel, c_badge = st.columns([3, 1])
    with c_sel:
        selected_key = st.selectbox(
            "Benchmark scenario:",
            list(rep_options.keys()),
            index=1,
            help="Representative test-period timestamp for instantaneous operating state.",
        )
    ts_str = rep_options[selected_key]
    ts = pd.to_datetime(ts_str)
    if ts not in sim_df.index:
        ts = sim_df.index[0]

    row = sim_df.loc[ts]
    gate_status = row["control_safety_gate"]

    with c_badge:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if gate_status == "PROTECTED_OCCUPIED":
            st.badge("PROTECTED OCCUPIED", color="orange", icon=":material/shield:")
        else:
            st.badge("SETBACK ELIGIBLE", color="green", icon=":material/check_circle:")

    prob_occ    = float(row["occupancy_probability"])
    occ_state   = int(row["occupancy_state"])
    outdoor_t   = float(row["outdoor_temp_c"])
    base_fan    = float(row["baseline_fan"])
    rec_fan     = float(row["recommended_fan"])
    base_damper = float(row["baseline_damper"])
    rec_damper  = float(row["recommended_damper"])
    base_energy = float(row["baseline_predicted_energy"])
    opt_energy  = float(row["optimized_predicted_energy"])
    delta_e     = float(row["optimized_delta_energy"])
    save_pct    = float(row["estimated_saving_percent"])
    cost_save   = float(row["estimated_cost_saving"])
    co2_save    = float(row["estimated_co2_reduction"])

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Forecasted occupancy",
        "OCCUPIED" if occ_state else "VACANT",
        f"P(occ) = {prob_occ * 100:.1f}%",
        delta_color="off",
    )
    c2.metric(
        "Outdoor temperature",
        f"{outdoor_t:.1f} \u00b0C",
        "Cold regime (<10 \u00b0C)" if outdoor_t < 10 else "Mild regime (\u226510 \u00b0C)",
        delta_color="inverse" if outdoor_t < 10 else "normal",
    )
    c3.metric(
        "Supply fan VFD",
        f"{rec_fan:.0f}%",
        f"{rec_fan - base_fan:+.0f}% vs baseline",
        delta_color="inverse",
    )
    c4.metric(
        "Damper outdoor air",
        f"{rec_damper:.0f}%",
        f"{rec_damper - base_damper:+.0f}% vs baseline",
        delta_color="inverse",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Baseline energy", f"{base_energy:.2f} kWh")
    c6.metric("Counterfactual energy", f"{opt_energy:.2f} kWh", f"-{save_pct:.1f}% reduction")
    c7.metric("Hourly savings", f"{-delta_e:.2f} kWh", f"{save_pct:.1f}% saved")
    c8.metric("Economic / carbon", f"${cost_save:.2f}/hr", f"-{co2_save:.2f} kg CO\u2082/hr")

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    _section(
        "Test-period trajectory (994 evaluated hours)",
        "Historical actual \u00b7 model baseline \u00b7 counterfactual optimized",
    )
    fig = _fig()
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["actual_total_energy"],
        mode="lines", name="Historical actual",
        line=dict(color=C["actual"], width=1, dash="dot"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["baseline_predicted_energy"],
        mode="lines", name="Model baseline",
        line=dict(color=C["baseline"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["optimized_predicted_energy"],
        mode="lines", name="Counterfactual optimized",
        line=dict(color=C["optimized"], width=2),
    ))
    fig.update_layout(height=360, xaxis_title="Simulation timestamp", yaxis_title="Hourly energy (kWh)")
    st.plotly_chart(fig, width="stretch")

    ca, cb = st.columns(2)
    with ca:
        st.success(f":material/lightbulb: **Optimizer recommendation:**\n\n{row['action_reason']}")
    with cb:
        st.warning(
            f":material/verified: **Surrogate validity:** `{row['confidence_flag']}`\n\n"
            f"Constraint status: `{row['constraint_status']}`"
        )


# =============================================================================
# PAGE 2 - SCENARIO SIMULATOR
# =============================================================================

def page_scenario_simulator(joint_df: pd.DataFrame, occ_art: Any, energy_art: Any):
    st.title(":material/tune: Interactive scenario simulator")
    _notice()
    st.caption(
        "Select a verified historical timestamp from the test set, optionally apply what-if overrides, "
        "then trigger the optimizer to explore counterfactual building states."
    )

    rep_options = get_representative_timestamps()
    c_mode, c_pick = st.columns([1, 2])
    with c_mode:
        sel_mode = st.radio("Selection mode:", ["Preset scenarios", "Custom timestamp"])
    with c_pick:
        if sel_mode == "Preset scenarios":
            preset = st.selectbox("Representative scenario:", list(rep_options.keys()))
            selected_ts = pd.to_datetime(rep_options[preset])
        else:
            all_ts = joint_df.index.tolist()
            idx = st.select_slider(
                "Timestamp from test period:",
                options=all_ts, value=all_ts[100],
                format_func=lambda x: str(x),
            )
            selected_ts = pd.to_datetime(idx)

    hist_row = joint_df.loc[selected_ts]
    if isinstance(hist_row, pd.DataFrame):
        hist_row = hist_row.iloc[0]

    _section(f"1. Sensor vector at `{selected_ts}`", "Real observed values from the test set.")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Indoor temp", f"{hist_row.get('indoor_temp_mean', 22.0):.1f} \u00b0C")
    s2.metric("Outdoor temp", f"{hist_row.get('outdoor_temp_c', 12.0):.1f} \u00b0C")
    s3.metric("Rel. humidity", f"{hist_row.get('relative_humidity', 50.0):.1f} %")
    s4.metric("Solar radiation", f"{hist_row.get('solar_radiation', 0.0):.0f} W/m\u00b2")
    s5.metric("Baseline fan", f"{hist_row.get('rtu_south_fan_spd_mean_next_hour', 65.0):.0f} %")
    s6.metric("Baseline damper", f"{hist_row.get('rtu_south_damper_pct_mean_next_hour', 30.0):.0f} %")

    with st.expander(":material/tune: What-if overrides (optional)", expanded=False):
        st.caption("Adjust inputs to simulate synthetic perturbations. Safety and ramp limits still enforced.")
        ov1, ov2, ov3, ov4 = st.columns(4)
        override_fan    = ov1.slider("Fan speed (%)", 40.0, 90.0,
                                     float(hist_row.get("rtu_south_fan_spd_mean_next_hour", 65.0)), 5.0)
        override_damper = ov2.slider("Damper (%)", 10.0, 90.0,
                                     float(hist_row.get("rtu_south_damper_pct_mean_next_hour", 30.0)), 5.0)
        override_tin    = ov3.slider("Indoor temp (\u00b0C)", 16.0, 28.0,
                                     float(hist_row.get("indoor_temp_mean", 22.0)), 0.5)
        override_tout   = ov4.slider("Outdoor temp (\u00b0C)", 2.0, 30.0,
                                     float(hist_row.get("outdoor_temp_c", 12.0)), 0.5)
        use_overrides   = st.checkbox("Apply overrides to optimization", value=False)

    overrides_dict = None
    if use_overrides:
        overrides_dict = {
            "rtu_south_fan_spd_mean_next_hour":    override_fan,
            "rtu_south_damper_pct_mean_next_hour": override_damper,
            "indoor_temp_mean":                    override_tin,
            "outdoor_temp_c":                      override_tout,
        }

    _section("2. Optimization trigger")
    st.button(":material/rocket_launch: Optimize building controls", type="primary", width="stretch")

    res = run_scenario_optimization(
        timestamp=selected_ts, joint_df=joint_df, occ_art=occ_art,
        energy_art=energy_art, overrides=overrides_dict,
        enforce_ramps=True, enforce_comfort=True, include_lighting=True,
    )

    _section("3. Recommended operating state & impact")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Supply fan VFD", f"{res['recommended_fan']:.0f}%",
              f"{res['recommended_fan'] - res['baseline_fan']:+.0f}% vs baseline", delta_color="inverse")
    r2.metric("Damper setpoint", f"{res['recommended_damper']:.0f}%",
              f"{res['recommended_damper'] - res['baseline_damper']:+.0f}% vs baseline", delta_color="inverse")
    r3.metric("Predicted energy", f"{res['optimized_predicted_energy']:.2f} kWh",
              f"\u0394E: {res['optimized_delta_energy']:+.2f} kWh ({res['estimated_saving_percent']:.1f}%)")
    r4.metric("Estimated savings", f"${res['estimated_cost_saving']:.2f}/hr",
              f"-{res['estimated_co2_reduction']:.2f} kg CO\u2082/hr")

    _section("4. Diagnostic attribution")
    att1, att2 = st.columns([2, 1])
    with att1:
        st.success(f":material/lightbulb: {res['action_reason']}")
        st.info(
            f":material/shield: **Safety gate:** `{res['control_safety_gate']}`\n\n"
            f"P(occ) = `{res['occupancy_probability'] * 100:.1f}%` "
            f"(T_class={T_CLASS}, T_safety={T_SAFETY})\n\n"
            f":material/verified: **Surrogate confidence:** `{res['confidence_flag']}`"
        )
    with att2:
        fig_bar = _fig()
        fig_bar.add_trace(go.Bar(
            name="Baseline",
            x=["Fan (%)", "Damper (%)", "Energy (kWh)"],
            y=[res["baseline_fan"], res["baseline_damper"], res["baseline_predicted_energy"]],
            marker_color=C["baseline"], marker_line_width=0,
        ))
        fig_bar.add_trace(go.Bar(
            name="Recommended",
            x=["Fan (%)", "Damper (%)", "Energy (kWh)"],
            y=[res["recommended_fan"], res["recommended_damper"], res["optimized_predicted_energy"]],
            marker_color=C["optimized"], marker_line_width=0,
        ))
        fig_bar.update_layout(barmode="group", height=240, margin=dict(l=10, r=10, t=10, b=30))
        st.plotly_chart(fig_bar, width="stretch")


# =============================================================================
# PAGE 3 - ENERGY ANALYTICS
# =============================================================================

def page_energy_analytics(sim_df: pd.DataFrame):
    st.title(":material/bolt: Energy analytics")
    _notice()

    _section("Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        date_range = st.date_input(
            "Date range:",
            value=[sim_df.index.min().date(), sim_df.index.max().date()],
            min_value=sim_df.index.min().date(), max_value=sim_df.index.max().date(),
        )
    with f2:
        occ_filter  = st.selectbox("Occupancy state:", ["All", "Occupied only", "Unoccupied only"])
    with f3:
        temp_filter = st.selectbox("Temperature regime:",
                                   ["All", "Cold (<10 \u00b0C)", "Mild (10-16 \u00b0C)", "Warm (>16 \u00b0C)"])

    f_df = sim_df.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s, e = date_range
        f_df = f_df.loc[(f_df.index.date >= s) & (f_df.index.date <= e)]
    if occ_filter == "Occupied only":
        f_df = f_df.loc[f_df["actual_occupancy_state"] == 1]
    elif occ_filter == "Unoccupied only":
        f_df = f_df.loc[f_df["actual_occupancy_state"] == 0]
    if temp_filter == "Cold (<10 \u00b0C)":
        f_df = f_df.loc[f_df["outdoor_temp_c"] < 10.0]
    elif temp_filter == "Mild (10-16 \u00b0C)":
        f_df = f_df.loc[(f_df["outdoor_temp_c"] >= 10.0) & (f_df["outdoor_temp_c"] <= 16.0)]
    elif temp_filter == "Warm (>16 \u00b0C)":
        f_df = f_df.loc[f_df["outdoor_temp_c"] > 16.0]

    st.caption(f"Showing **{len(f_df):,}** filtered hours out of **{len(sim_df):,}** total test hours.")

    _section("1. Energy trajectory: actual vs baseline vs counterfactual")
    fig1 = _fig()
    fig1.add_trace(go.Scatter(x=f_df.index, y=f_df["actual_total_energy"],
                              mode="lines", name="Historical actual",
                              line=dict(color=C["actual"], width=1, dash="dot"), opacity=0.7))
    fig1.add_trace(go.Scatter(x=f_df.index, y=f_df["baseline_predicted_energy"],
                              mode="lines", name="Model baseline",
                              line=dict(color=C["baseline"], width=1.5)))
    fig1.add_trace(go.Scatter(x=f_df.index, y=f_df["optimized_predicted_energy"],
                              mode="lines", name="Counterfactual optimized",
                              line=dict(color=C["optimized"], width=2)))
    fig1.update_layout(height=360, xaxis_title="Timestamp", yaxis_title="Hourly energy (kWh)")
    st.plotly_chart(fig1, width="stretch")

    c2, c3 = st.columns(2)
    with c2:
        _section("2. Actual vs baseline prediction")
        fig2 = px.scatter(
            f_df, x="actual_total_energy", y="baseline_predicted_energy",
            color="outdoor_temp_c", color_continuous_scale="Blues",
            labels={"actual_total_energy": "Actual (kWh)",
                    "baseline_predicted_energy": "Baseline pred (kWh)",
                    "outdoor_temp_c": "Outdoor T (\u00b0C)"},
            template=_TMPL, height=320,
        )
        _mx = float(f_df["actual_total_energy"].max() or 30)
        fig2.add_shape(type="line", line=dict(dash="dash", color=C["danger"], width=1.5),
                       x0=0, y0=0, x1=_mx, y1=_mx)
        fig2.update_layout(**_layout_extras())
        st.plotly_chart(fig2, width="stretch")

    with c3:
        _section("3. Actual vs counterfactual")
        fig3 = px.scatter(
            f_df, x="actual_total_energy", y="optimized_predicted_energy",
            color="outdoor_temp_c", color_continuous_scale="Greens",
            labels={"actual_total_energy": "Actual (kWh)",
                    "optimized_predicted_energy": "Optimized pred (kWh)",
                    "outdoor_temp_c": "Outdoor T (\u00b0C)"},
            template=_TMPL, height=320,
        )
        fig3.update_layout(**_layout_extras())
        st.plotly_chart(fig3, width="stretch")

    c4, c5 = st.columns(2)
    with c4:
        _section("4. Hourly \u0394E (estimated savings)")
        fig4 = _fig()
        fig4.add_trace(go.Bar(x=f_df.index, y=f_df["optimized_delta_energy"],
                              name="\u0394E = E_cand \u2212 E_base",
                              marker_color=C["optimized"], marker_line_width=0))
        fig4.update_layout(height=320, xaxis_title="Timestamp", yaxis_title="Estimated \u0394E (kWh)",
                            yaxis=dict(zeroline=True, zerolinecolor=C["danger"], gridcolor="#1F2E44"))
        st.plotly_chart(fig4, width="stretch")

    with c5:
        _section("5. Cumulative estimated energy savings")
        cum = (-f_df["optimized_delta_energy"]).cumsum()
        fig5 = _fig()
        fig5.add_trace(go.Scatter(x=f_df.index, y=cum, mode="lines",
                                   name="Cumulative savings (kWh)",
                                   line=dict(color=C["savings"], width=2.5),
                                   fill="tozeroy", fillcolor="rgba(56,189,248,0.12)"))
        fig5.update_layout(height=320, xaxis_title="Timestamp", yaxis_title="Cumulative saved kWh")
        st.plotly_chart(fig5, width="stretch")

    c6, c7 = st.columns(2)
    with c6:
        _section("6. Hourly savings distribution")
        fig6 = px.histogram(
            f_df.loc[f_df["estimated_energy_saving"] > 0],
            x="estimated_energy_saving", nbins=35,
            template=_TMPL, height=320,
            labels={"estimated_energy_saving": "Estimated saving (kWh)"},
            color_discrete_sequence=[C["optimized"]],
        )
        fig6.update_layout(**_layout_extras(),
                           xaxis_title="Hourly saving (kWh)", yaxis_title="Frequency (hours)")
        st.plotly_chart(fig6, width="stretch")

    with c7:
        _section("7. South wing submeter breakdown (actual)")
        if "hvac_actual_kwh" in f_df.columns:
            sub = pd.DataFrame({
                "Submeter": ["HVAC (RTU South)", "Lighting", "Plug loads (MELs)"],
                "Total energy (kWh)": [
                    f_df["hvac_actual_kwh"].sum(),
                    f_df["lig_actual_kwh"].sum(),
                    f_df["mels_actual_kwh"].sum(),
                ],
            })
            fig7 = px.pie(sub, values="Total energy (kWh)", names="Submeter",
                          color_discrete_sequence=[C["savings"], C["baseline"], "#EC4899"],
                          template=_TMPL, hole=0.45, height=320)
            fig7.update_traces(textfont_size=13)
            fig7.update_layout(**_layout_extras())
            st.plotly_chart(fig7, width="stretch")
        else:
            st.caption("Submeter breakdown not available for this filter selection.")


# =============================================================================
# PAGE 4 - OCCUPANCY ANALYTICS
# =============================================================================

def page_occupancy_analytics(sim_df: pd.DataFrame):
    st.title(":material/group: Occupancy analytics")
    _notice()
    st.caption(
        "The South Wing occupancy forecaster (t\u2192t+1) acts as the master safety barrier for setback controls."
    )

    _section("1. Occupancy probability trajectory & dual safety thresholds")
    fig_occ = _fig()
    fig_occ.add_trace(go.Scatter(x=sim_df.index, y=sim_df["occupancy_probability"],
                                  mode="lines", name="P(Occ=1) \u2014 predicted",
                                  line=dict(color=C["savings"], width=1.5)))
    fig_occ.add_trace(go.Scatter(x=sim_df.index, y=sim_df["actual_occupancy_state"],
                                  mode="lines", name="Actual ground-truth",
                                  line=dict(color=C["actual"], width=1, dash="dot"), opacity=0.6))
    fig_occ.add_hline(y=T_CLASS, line_dash="dash", line_color=C["baseline"],
                       annotation_text=f"T_class = {T_CLASS:.2f} (optimal decision threshold)",
                       annotation_position="top left")
    fig_occ.add_hline(y=T_SAFETY, line_dash="dash", line_color=C["danger"],
                       annotation_text=f"T_safety = {T_SAFETY:.2f} (conservative safety gate)",
                       annotation_position="bottom left")
    fig_occ.update_layout(height=360, xaxis_title="Timestamp", yaxis_title="Probability [0, 1]")
    st.plotly_chart(fig_occ, width="stretch")

    cd1, cd2 = st.columns(2)
    with cd1:
        _section("2. Diurnal occupancy profile by day type")
        df_p = sim_df.copy()
        df_p["hour"] = df_p.index.hour
        df_p["is_weekend"] = df_p.index.dayofweek >= 5
        diurnal = df_p.groupby(["hour", "is_weekend"])["actual_occupancy_state"].mean().reset_index()
        diurnal["Day type"] = diurnal["is_weekend"].map({True: "Weekend", False: "Weekday"})
        fig_d = px.line(diurnal, x="hour", y="actual_occupancy_state", color="Day type",
                        template=_TMPL, markers=True, height=320,
                        labels={"hour": "Hour of day", "actual_occupancy_state": "Mean occupancy rate"},
                        color_discrete_map={"Weekday": C["savings"], "Weekend": C["baseline"]})
        fig_d.update_layout(**_layout_extras())
        st.plotly_chart(fig_d, width="stretch")

    with cd2:
        _section("3. Weekly occupancy heatmap")
        df_h = sim_df.copy()
        df_h["day_name"] = df_h.index.day_name()
        df_h["hour"]     = df_h.index.hour
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat_m = df_h.groupby(["day_name", "hour"])["occupancy_probability"].mean().unstack()
        heat_m = heat_m.reindex(day_order)
        fig_h = px.imshow(heat_m, template=_TMPL, height=320,
                          labels=dict(x="Hour of day", y="Day of week", color="Mean P(Occ)"),
                          color_continuous_scale="Blues")
        fig_h.update_layout(**_layout_extras())
        st.plotly_chart(fig_h, width="stretch")

    _section("4. Champion classifier \u2014 frozen test metrics")
    p3_res  = load_benchmark_metrics()
    occ_t   = p3_res.get("occupancy_classification", {}).get("test_results", {})
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("F1 (T=0.51)",     f"{occ_t.get('Test F1 (Tuned)', 0.9419):.4f}")
    m2.metric("Precision",       f"{occ_t.get('Test Precision', 0.9332):.4f}")
    m3.metric("Recall",          f"{occ_t.get('Test Recall', 0.9508):.4f}")
    m4.metric("ROC-AUC",         f"{occ_t.get('ROC-AUC', 0.9392):.4f}")
    m5.metric("PR-AUC",          f"{occ_t.get('PR-AUC', 0.9806):.4f}")
    m6.metric("Balanced acc.",   f"{occ_t.get('Bal Acc', 0.8643):.4f}")

    st.info(
        ":material/info: **Dual-threshold safety architecture** \u2014 "
        "**T_class=0.51** (val-tuned, max F1=0.9419) is used for reporting. "
        "**T_safety=0.30** is a deliberately conservative operational guardrail: "
        "any predicted P(occ) \u2265 0.30 marks the space `PROTECTED_OCCUPIED`, "
        "reducing false-setback occupied risk hours by **72.55%** during testing."
    )


# =============================================================================
# PAGE 5 - ML MODEL PERFORMANCE
# =============================================================================

def page_model_performance():
    st.title(":material/psychology: ML model benchmark & audit")
    _notice()
    st.caption("Frozen Phase 3 candidate benchmarks and Phase 3.5 robustness audits. No models are retrained here.")

    p3    = load_benchmark_metrics()
    p3rob = load_robustness_metrics()

    tab_occ, tab_eng, tab_base = st.tabs([
        ":material/group: Occupancy models",
        ":material/bolt: Energy surrogates",
        ":material/analytics: Naive baselines",
    ])

    with tab_occ:
        _section("Candidate occupancy model benchmark (validation set)")
        val_occ = p3.get("occupancy_classification", {}).get("validation_results", [])
        if val_occ:
            df_v = pd.DataFrame(val_occ)
            st.dataframe(df_v, width="stretch", hide_index=True)
            fig = px.bar(df_v, x="Model", y="Val F1 (Tuned)", color="PR-AUC",
                         template=_TMPL, height=320,
                         title="Validation F1-score across architectures",
                         color_continuous_scale="Blues")
            fig.update_layout(**_layout_extras())
            st.plotly_chart(fig, width="stretch")
        _section("Frozen test set \u2014 Random Forest champion")
        st.json(p3.get("occupancy_classification", {}).get("test_results", {}), expanded=False)

    with tab_eng:
        _section("Candidate energy model benchmark (validation set)")
        val_eng = p3.get("energy_forecasting", {}).get("validation_results", [])
        if val_eng:
            df_e = pd.DataFrame(val_eng)
            st.dataframe(df_e, width="stretch", hide_index=True)
            ycol = "Chained RMSE" if "Chained RMSE" in df_e.columns else df_e.columns[2]
            fig_e = px.bar(df_e, x="Model", y=ycol, template=_TMPL, height=320,
                           title="Validation RMSE across energy model candidates",
                           color_discrete_sequence=[C["baseline"]])
            fig_e.update_layout(**_layout_extras())
            st.plotly_chart(fig_e, width="stretch")
        _section("Frozen test set \u2014 Ridge Regression champion (\u03b1=10)")
        st.json(p3.get("energy_forecasting", {}).get("final_test_metrics_chained", {}), expanded=False)
        st.info(
            ":material/info: **Architecture rationale:** Tree ensembles exhibited non-physical step-function "
            "artifacts and extrapolation under counterfactual setpoints. **Ridge Regression (\u03b1=10)** "
            "was selected: linear gradients maintain monotonicity and predictable sensitivity."
        )

    with tab_base:
        _section("Naive forecasting baseline audit (Phase 3.5)")
        st.caption("Candidate ML models audited against simple baselines to prevent overconfidence.")
        for k, v in p3rob.get("baselines_summary", {}).items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")
        _section("Temperature regime stratification (test period)")
        regimes = p3rob.get("regime_analysis", {}).get("outdoor_temperature", [])
        if regimes:
            st.dataframe(pd.DataFrame(regimes), width="stretch", hide_index=True)


# =============================================================================
# PAGE 6 - OPTIMIZATION ANALYSIS
# =============================================================================

def page_optimization_analysis(sim_df: pd.DataFrame):
    st.title(":material/track_changes: Counterfactual optimization analysis")
    _notice()

    p4   = load_phase4_summary()
    kpis = p4.get("overall_simulation_metrics", {})

    _section("Test-period simulation aggregate KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Evaluated hours",  f"{kpis.get('hours_evaluated', 994):,} hrs")
    k2.metric("Hours optimized",  f"{kpis.get('hours_optimized', 804):,} hrs",
              f"{kpis.get('pct_hours_improved', 80.89):.1f}% active")
    k3.metric("Estimated energy savings",
              f"{kpis.get('total_estimated_energy_savings_kwh', 5120.45):,.1f} kWh",
              f"{kpis.get('overall_estimated_energy_savings_pct', 53.32):.1f}% reduction")
    k4.metric("Estimated cost savings",
              f"${kpis.get('total_estimated_cost_savings_usd', 1126.50):,.2f}",
              f"-{kpis.get('total_estimated_co2_reduction_kg', 1075.29):,.1f} kg CO\u2082 avoided")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Baseline preserved (NO_CHANGE)", f"{kpis.get('hours_no_change', 190):,} hrs")
    k6.metric("Mean \u0394E / optimized hour",
              f"{kpis.get('mean_delta_per_optimized_hour_kwh', -6.37):.2f} kWh")
    k7.metric("Median \u0394E / optimized hour",
              f"{kpis.get('median_delta_per_optimized_hour_kwh', -6.58):.2f} kWh")
    k8.metric("CO\u2082 avoided",
              f"{kpis.get('total_estimated_co2_reduction_kg', 1075.29):,.1f} kg")

    _section("Regime stratification breakdown")
    rd = p4.get("regime_breakdown", {})
    rb1, rb2 = st.columns(2)
    with rb1:
        _section("By occupancy regime")
        df_ob = pd.DataFrame([rd.get("occupancy_occupied", {}), rd.get("occupancy_unoccupied", {})])
        if not df_ob.empty:
            fig_ob = px.bar(df_ob, x="regime", y="savings_kwh", color="regime",
                            template=_TMPL, height=280,
                            labels={"regime": "Occupancy regime", "savings_kwh": "Estimated savings (kWh)"},
                            color_discrete_map={"Occupied": C["occupied"], "Unoccupied": C["optimized"]})
            fig_ob.update_layout(**_layout_extras())
            st.plotly_chart(fig_ob, width="stretch")
    with rb2:
        _section("By temperature regime")
        df_tb = pd.DataFrame([rd.get("temp_cold", {}), rd.get("temp_mild", {}), rd.get("temp_warm", {})])
        if not df_tb.empty:
            fig_tb = px.bar(df_tb, x="regime", y="savings_kwh", color="regime",
                            template=_TMPL, height=280,
                            labels={"regime": "Temperature regime", "savings_kwh": "Estimated savings (kWh)"},
                            color_discrete_sequence=[C["savings"], C["optimized"], C["baseline"]])
            fig_tb.update_layout(**_layout_extras())
            st.plotly_chart(fig_tb, width="stretch")

    _section("Safeguard ablation study")
    abl_rows = [
        {
            "Ablation scenario":   k,
            "Hours optimized":     v.get("hours_optimized"),
            "Savings (kWh)":       v.get("total_estimated_energy_savings_kwh"),
            "Saving %":            v.get("overall_estimated_energy_savings_pct"),
            "Cost savings ($)":    v.get("total_estimated_cost_savings_usd"),
        }
        for k, v in p4.get("ablations", {}).items()
    ]
    st.dataframe(pd.DataFrame(abl_rows), width="stretch", hide_index=True)
    st.error(
        ":material/warning: **Critical callout \u2014 the non-physical no-ramp ablation:** "
        "Ablation C (no ramp protection) reports **115.86% estimated savings (11,125 kWh)** \u2014 "
        "a mathematically implausible figure caused by unconstrained 50% instant fan-speed jumps. "
        "The Full Safeguards Policy restricts fan ramp to **\u226415%/hr**, "
        "producing the realistic **53.32%** reduction."
    )


# =============================================================================
# PAGE 7 - SAFETY & VALIDITY
# =============================================================================

def page_safety_validity():
    st.title(":material/shield: Safety constraints & domain validity")
    _notice()
    st.caption(
        "Because ML models cannot intrinsically understand fluid mechanics or building acoustics, "
        "the optimizer is wrapped in a multi-layered deterministic safety envelope."
    )

    _section("Operational safety bounds envelope")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Fan speed bounds",   "40 \u2013 90 %",    "Historical support")
    b2.metric("Damper bounds",      "10 \u2013 90 %",    "Ventilation floor")
    b3.metric("Max fan ramp",       "\u226415 %/hr",     "Equipment protection")
    b4.metric("Max damper ramp",    "\u226430 %/hr",     "Actuator protection")
    b5, b6, b7, b8 = st.columns(4)
    b5.metric("Safety threshold T_safety",  f"{T_SAFETY:.2f}",              "Occupancy setback guard")
    b6.metric("Improvement epsilon \u03b5", f"{MIN_IMPROVEMENT_EPSILON:.2f} kWh", "Prevents hunting")
    b7.metric("Energy display floor",       f"{BASELOAD_ENERGY_FLOOR:.2f} kWh",  "Baseload preservation")
    b8.metric("Candidate grid",             "99 pairs",                          "Fan\u00d7Damper discrete search")

    _section("The four system operational states")
    s1, s2 = st.columns(2)
    with s1:
        st.success(
            ":material/check_circle: **CONTROL-SAFE (SETBACK ELIGIBLE)**\n\n"
            "P(occ) strictly below T_safety = 0.30. Zone confirmed vacant. "
            "Optimizer authorized to evaluate minimal fan speeds (down to 40%) and reduced dampers."
        )
        st.warning(
            ":material/shield: **PROTECTED_OCCUPIED**\n\n"
            "P(occ) \u2265 0.30. Deep setback prohibited regardless of occupancy level. "
            "Fan speeds must maintain adequate ventilation and thermal comfort."
        )
    with s2:
        st.error(
            ":material/thermostat: **LOW_SURROGATE_CONFIDENCE (cold regime <10 \u00b0C)**\n\n"
            "Outdoor temperature below 10 \u00b0C. Cold weather was virtually absent in training data "
            "(May\u2013Nov 2018). Linear surrogate exhibits negative bias; tagged low-confidence."
        )
        st.info(
            ":material/do_not_disturb: **NO_CHANGE (deadband preservation)**\n\n"
            "If no candidate setpoint yields improvement \u2265 \u03b5 = 0.20 kWh, the optimizer "
            "issues NO_CHANGE to preserve existing states and prevent valve hunting."
        )

    _section("Domain limitations & scope boundaries")
    st.warning(
        ":material/warning: **Empirical surrogate vs physics simulator:** "
        "The Ridge model is an empirical regression surface, not EnergyPlus. "
        "It models consumption but does not co-simulate zone thermal capacitance or reheat loops.\n\n"
        ":material/thermostat: **Indoor temperature dynamics:** "
        "Adjusting supply fan setpoints does not update future indoor temperatures within this simulation.\n\n"
        ":material/signal_disconnected: **No closed-loop actuation:** "
        "This is an offline academic research simulation. No BACnet, Modbus, or MQTT signals are dispatched."
    )


# =============================================================================
# PAGE 8 - ABOUT & METHODOLOGY
# =============================================================================

def page_about_methodology():
    st.title(":material/menu_book: About & modeling methodology")
    _notice()

    st.markdown(
        "This interactive platform encapsulates the complete end-to-end BEMS research workflow for "
        "**LBNL Building 59 (South Wing, Floors 3 & 4)** \u2014 a high-density commercial office environment."
    )

    _section("System pipeline")
    st.code(
        "Building sensor state at time t\n"
        "         |\n"
        "[Occupancy Forecast t->t+1]  Random Forest, T_class=0.51, T_safety=0.30\n"
        "         |\n"
        "[Energy Forecast t->t+1]     Ridge Regression, alpha=10\n"
        "         |\n"
        "[Counterfactual Optimizer]   Bounded grid search, 99 (fan x damper) pairs\n"
        "         |                   Ramp limits + occupancy safety gate\n"
        "Recommended setpoints + DeltaE = E_candidate - E_baseline",
        language=None,
    )

    _section("Technical specifications")
    st.markdown(
        "| Dimension | Specification | Notes |\n"
        "| :--- | :--- | :--- |\n"
        "| **Dataset** | LBNL Building 59 | Open-source multi-sensor commercial building |\n"
        "| **Spatial scope** | South Wing (Floors 3 & 4) | High sensor-density office + workspace |\n"
        "| **Time resolution** | 1 hour | Leakage-safe backward-looking features |\n"
        "| **Prediction horizon** | t \u2192 t+1 | Features at t forecast conditions at t+1 |\n"
        "| **Occupancy model** | Random Forest | T_class=0.51, T_safety=0.30 |\n"
        "| **Energy model** | Ridge (\u03b1=10) | Monotonic gradient, robust CF extrapolation |\n"
        "| **Candidate space** | 99 pairs | Fan {40,45,\u2026,90}% \u00d7 Damper {10,20,\u2026,90}% |\n"
        "| **Electricity tariff** | $0.22/kWh | PG&E Commercial B-19 schedule |\n"
        "| **Carbon intensity** | 0.210 kg CO\u2082/kWh | EPA eGRID CAMX subregion |\n"
        "| **Display energy floor** | 6.50 kWh | Calibrated baseload display floor |\n"
    )

    _section("Mathematical formulation")
    st.latex(
        r"\Delta E(u) = \hat{E}(u_{\text{candidate}}, \hat{O}_{t+1}, X_t)"
        r" - \hat{E}(u_{\text{baseline}}, \hat{O}_{t+1}, X_t)"
    )
    st.markdown(
        r"Where $u = (\text{fan\_spd}, \text{damper\_pct})$ is the controllable HVAC actuator vector, "
        r"$\hat{O}_{t+1} \in \{0,1\}$ is the predicted occupancy state, "
        r"$X_t$ is the frozen exogenous context. "
        r"A **negative** $\Delta E$ indicates estimated energy reduction."
    )

    _section("Citation & attribution")
    st.markdown(
        "- **Dataset:** *Building 59 Multi-Sensor Dataset*, Lawrence Berkeley National Laboratory (LBNL)\n"
        "- **Pipeline:** Academic BEMS Energy Optimization \u2014 Phases 0\u20135 (2026)"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    selected = render_sidebar()

    try:
        sim_df = load_simulation_results()
        joint_df = load_joint_test_data()
        occ_art, energy_art = get_cached_models()
    except Exception as exc:
        st.error(f":material/error: Failed to initialize dashboard resources: {exc}")
        st.info("Ensure all preprocessing and training phases have completed and models are serialized.")
        return

    dispatch = {
        "Executive Overview":    lambda: page_executive_overview(sim_df),
        "Scenario Simulator":    lambda: page_scenario_simulator(joint_df, occ_art, energy_art),
        "Energy Analytics":      lambda: page_energy_analytics(sim_df),
        "Occupancy Analytics":   lambda: page_occupancy_analytics(sim_df),
        "ML Model Performance":  page_model_performance,
        "Optimization Analysis": lambda: page_optimization_analysis(sim_df),
        "Safety & Validity":     page_safety_validity,
        "About & Methodology":   page_about_methodology,
    }
    dispatch[selected]()


if __name__ == "__main__":
    main()
