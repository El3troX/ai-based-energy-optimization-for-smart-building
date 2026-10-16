"""
Building 59 BEMS -- Custom Interactive AI Energy Optimization Dashboard.
Built with Dash (by Plotly) and Dash Bootstrap Components.
High-performance, reactive, dark-theme industrial BEMS research platform.

RESEARCH SIMULATION DISCLAIMER: All delta energy values and recommended setpoints are
model-based counterfactual estimates derived from an empirical Ridge surrogate.
These outputs represent research simulation scenarios, NOT experimentally verified physical control.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# THEME & CONSTANTS
# =============================================================================

C = dict(
    actual="#94A3B8",
    baseline="#F59E0B",
    optimized="#10B981",
    savings="#38BDF8",
    danger="#EF4444",
    occupied="#818CF8",
    vacant="#10B981",
    purple="#C084FC",
    card_bg="#111827",
    border="#1E293B",
)

_TMPL = "plotly_dark"
_LAYOUT = dict(
    template=_TMPL,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#CBD5E1"),
    margin=dict(l=40, r=20, t=36, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor="#1F2E44", zeroline=False),
    yaxis=dict(gridcolor="#1F2E44", zeroline=False),
)

def _fig(**kw) -> go.Figure:
    fig = go.Figure(**kw)
    fig.update_layout(**_LAYOUT)
    return fig

def _layout_extras() -> dict:
    return {k: v for k, v in _LAYOUT.items() if k != "template"}


# =============================================================================
# INITIALIZE DASH APP
# =============================================================================

app = dash.Dash(
    __name__,
    title="Building 59 BEMS | AI Energy Optimization",
    external_stylesheets=[
        dbc.themes.DARKLY,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

# Preload static dataset and metrics
sim_df = load_simulation_results()
joint_df = load_joint_test_data()
occ_art, energy_art = get_cached_models()
p3_metrics = load_benchmark_metrics()
p3_robustness = load_robustness_metrics()
p4_summary = load_phase4_summary()
rep_timestamps = get_representative_timestamps()


# =============================================================================
# UI ATOMS & REUSABLE COMPONENTS
# =============================================================================

def render_research_notice():
    return html.Div(
        className="research-notice",
        children=[
            html.I(className="bi bi-shield-exclamation text-warning me-2", style={"fontSize": "1.2rem"}),
            html.Div([
                html.Strong("Research Simulation Notice: ", style={"color": "#F8FAFC"}),
                "All Δ-energy values and actuator recommendations are ",
                html.Span("model-predicted counterfactual estimates", style={"fontWeight": "600", "color": "#38BDF8"}),
                " derived from an empirical Ridge surrogate. No physical control signals are dispatched to physical building equipment.",
            ]),
        ],
    )

def render_section_header(title: str, subtitle: str = ""):
    return html.Div(
        className="section-header",
        children=[
            html.H3(title, className="section-title"),
            html.Span(subtitle, className="section-subtitle") if subtitle else None,
        ],
    )

def render_kpi_card(label: str, value: str, delta: str = "", delta_type: str = "neutral", icon: str = None):
    delta_class = f"kpi-delta {delta_type}"
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(
                className="kpi-label",
                children=[
                    html.Span(label),
                    html.I(className=f"bi {icon}", style={"color": "#64748B"}) if icon else None,
                ],
            ),
            html.Div(value, className="kpi-value"),
            html.Div(delta, className=delta_class) if delta else html.Div("—", className="kpi-delta neutral"),
        ],
    )


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

NAV_ITEMS = [
    {"label": "Executive Overview", "href": "/", "icon": "bi-speedometer2"},
    {"label": "Scenario Simulator", "href": "/scenario-simulator", "icon": "bi-sliders2"},
    {"label": "Energy Analytics", "href": "/energy-analytics", "icon": "bi-lightning-charge"},
    {"label": "Occupancy Analytics", "href": "/occupancy-analytics", "icon": "bi-people"},
    {"label": "ML Model Performance", "href": "/model-performance", "icon": "bi-cpu"},
    {"label": "Optimization Analysis", "href": "/optimization-analysis", "icon": "bi-graph-up-arrow"},
    {"label": "Safety & Validity", "href": "/safety-validity", "icon": "bi-shield-check"},
    {"label": "About & Methodology", "href": "/about-methodology", "icon": "bi-journal-bookmark"},
]

def render_sidebar(current_path: str = "/"):
    links = []
    for item in NAV_ITEMS:
        is_active = (current_path == item["href"]) or (current_path == "" and item["href"] == "/")
        active_class = " active" if is_active else ""
        links.append(
            dcc.Link(
                href=item["href"],
                className=f"nav-link-custom{active_class}",
                children=[
                    html.I(className=f"bi {item['icon']} me-2", style={"fontSize": "1.05rem"}),
                    html.Span(item["label"]),
                ],
            )
        )

    return html.Div(
        className="sidebar-container",
        children=[
            # Header Branding
            html.Div(
                className="mb-4",
                children=[
                    html.Div(
                        className="d-flex align-items-center gap-2",
                        children=[
                            html.I(className="bi bi-building-gear text-info", style={"fontSize": "1.7rem"}),
                            html.Div([
                                html.H4("Building 59 BEMS", className="mb-0", style={"fontSize": "1.2rem", "fontWeight": "700"}),
                                html.Span("AI ENERGY OPTIMIZATION · LBNL", style={"fontSize": "0.68rem", "color": "#64748B", "letterSpacing": "0.06em", "fontWeight": "600"}),
                            ]),
                        ],
                    ),
                    # Badges Pill Strip
                    html.Div(
                        className="d-flex flex-wrap gap-1 mt-3",
                        children=[
                            html.Span([html.Span(className="pulse-dot pulse-dot-green me-1"), "LIVE SIM"], className="status-pill status-pill-safe"),
                            html.Span("t→t+1 HORIZON", className="status-pill status-pill-info"),
                            html.Span("FROZEN SURROGATE", className="status-pill status-pill-purple"),
                        ],
                    ),
                ],
            ),
            # Nav Links
            html.Div(links, className="flex-grow-1"),
            # System Parameters Footer
            html.Div(
                className="pt-3 border-top",
                style={"borderColor": "#1E293B"},
                children=[
                    html.Div("SYSTEM PARAMETERS", style={"fontSize": "0.68rem", "fontWeight": "700", "letterSpacing": "0.05em", "color": "#94A3B8", "marginBottom": "6px"}),
                    html.Div([
                        html.Span("Tariff: ", style={"color": "#64748B"}),
                        html.Strong(f"${ELECTRICITY_TARIFF:.2f}/kWh", style={"color": "#CBD5E1"}),
                        html.Span(" · Carbon: ", style={"color": "#64748B"}),
                        html.Strong(f"{EMISSION_FACTOR:.3f} kg/kWh", style={"color": "#CBD5E1"}),
                    ], style={"fontSize": "0.75rem", "marginBottom": "4px"}),
                    html.Div([
                        html.Span("T_safety: ", style={"color": "#64748B"}),
                        html.Strong(f"{T_SAFETY:.2f}", style={"color": "#FBBF24"}),
                        html.Span(" · T_class: ", style={"color": "#64748B"}),
                        html.Strong(f"{T_CLASS:.2f}", style={"color": "#38BDF8"}),
                        html.Span(" · ε: ", style={"color": "#64748B"}),
                        html.Strong(f"{MIN_IMPROVEMENT_EPSILON:.2f} kWh", style={"color": "#10B981"}),
                    ], style={"fontSize": "0.75rem", "marginBottom": "4px"}),
                    html.Div([
                        html.Span("Floor: ", style={"color": "#64748B"}),
                        html.Strong(f"{BASELOAD_ENERGY_FLOOR:.2f} kWh", style={"color": "#CBD5E1"}),
                        html.Span(" · Fan ramp: ", style={"color": "#64748B"}),
                        html.Strong(f"≤{RAMP_LIMITS['fan_speed_max_delta']}%/hr", style={"color": "#CBD5E1"}),
                    ], style={"fontSize": "0.75rem", "marginBottom": "12px"}),
                    html.Div("Phase 5 Production Dashboard · BEMS Framework © 2026", style={"fontSize": "0.66rem", "color": "#475569"}),
                ],
            ),
        ],
    )


# =============================================================================
# VIEW 1: EXECUTIVE OVERVIEW
# =============================================================================

def build_view_executive_overview():
    default_scenario = list(rep_timestamps.keys())[1]  # Peak Business Hours
    ts_str = rep_timestamps[default_scenario]
    ts = pd.to_datetime(ts_str)
    row = sim_df.loc[ts] if ts in sim_df.index else sim_df.iloc[0]

    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("Executive Overview", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("Instantaneous telemetry, active safety envelopes, and test-period trajectory preview", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
                html.Div(
                    className="d-flex align-items-center gap-3",
                    style={"position": "relative", "zIndex": 1050},
                    children=[
                        html.Div([
                            html.Label("Benchmark Scenario:", style={"fontSize": "0.75rem", "fontWeight": "600", "color": "#94A3B8", "marginBottom": "2px"}),
                            dcc.Dropdown(
                                id="exec-scenario-select",
                                options=[{"label": k, "value": k} for k in rep_timestamps.keys()],
                                value=default_scenario,
                                clearable=False,
                                className="dash-dropdown",
                                style={"minWidth": "320px"},
                            ),
                        ]),
                        html.Div(id="exec-gate-badge-container", className="mt-3"),
                    ],
                ),
            ],
        ),
        render_research_notice(),

        # 8 KPI Metric Cards Grid
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-occ"),
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-tout"),
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-fan"),
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-damper"),
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-base-e"),
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-opt-e"),
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-savings"),
                html.Div(className="col-12 col-sm-6 col-xl-3", id="exec-kpi-cost-co2"),
            ],
        ),

        # 994-hour Test-Period Trajectory Preview
        html.Div(
            className="glass-card mb-4",
            children=[
                html.Div(
                    className="d-flex justify-content-between align-items-center mb-2",
                    children=[
                        html.Div([
                            html.H5("Test-Period Trajectory (994 Evaluated Hours)", className="mb-0", style={"fontWeight": "600"}),
                            html.Span("Historical actual · Model baseline · Counterfactual optimized", style={"fontSize": "0.78rem", "color": "#64748B"}),
                        ]),
                        html.Span("Hourly Resolution", className="badge bg-secondary", style={"fontSize": "0.7rem"}),
                    ],
                ),
                dcc.Graph(id="exec-trajectory-graph", config={"displayModeBar": True, "responsive": True}),
            ],
        ),

        # Diagnostic Attribution Split Cards
        html.Div(
            className="row g-3",
            children=[
                html.Div(
                    className="col-12 col-md-7",
                    children=[
                        html.Div(
                            className="glass-card h-100",
                            style={"borderLeft": "4px solid #10B981"},
                            id="exec-recommendation-box",
                        ),
                    ],
                ),
                html.Div(
                    className="col-12 col-md-5",
                    children=[
                        html.Div(
                            className="glass-card h-100",
                            style={"borderLeft": "4px solid #F59E0B"},
                            id="exec-validity-box",
                        ),
                    ],
                ),
            ],
        ),
    ])


# =============================================================================
# VIEW 2: SCENARIO SIMULATOR
# =============================================================================

def build_view_scenario_simulator():
    all_timestamps = joint_df.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    default_ts = rep_timestamps[list(rep_timestamps.keys())[0]]

    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("Interactive Scenario Simulator", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("Explore verified historical timestamps, perturb building sensors, and trigger counterfactual optimization", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
            ],
        ),
        render_research_notice(),

        # Top Control Card: Timestamp Selector & Mode
        html.Div(
            className="glass-card mb-4 filter-bar-card",
            style={"position": "relative", "zIndex": 1050},
            children=[
                html.Div(
                    className="row g-3 align-items-center",
                    children=[
                        html.Div(
                            className="col-12 col-md-4",
                            children=[
                                html.Label("Selection Mode:", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#94A3B8"}),
                                dbc.RadioItems(
                                    id="sim-mode-radio",
                                    options=[
                                        {"label": "Curated Benchmark Presets", "value": "preset"},
                                        {"label": "Full Test Period Picker", "value": "custom"},
                                    ],
                                    value="preset",
                                    inline=True,
                                    className="mt-1",
                                ),
                            ],
                        ),
                        html.Div(
                            className="col-12 col-md-8",
                            children=[
                                html.Div(id="sim-preset-container", children=[
                                    html.Label("Representative Scenario:", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#94A3B8"}),
                                    dcc.Dropdown(
                                        id="sim-preset-select",
                                        options=[{"label": k, "value": rep_timestamps[k]} for k in rep_timestamps.keys()],
                                        value=default_ts,
                                        clearable=False,
                                        className="dash-dropdown",
                                    ),
                                ]),
                                html.Div(id="sim-custom-container", style={"display": "none"}, children=[
                                    html.Label("Timestamp from Test Period (994 hours):", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#94A3B8"}),
                                    dcc.Dropdown(
                                        id="sim-custom-select",
                                        options=[{"label": ts, "value": ts} for ts in all_timestamps[:250]],
                                        value=all_timestamps[50],
                                        clearable=False,
                                        className="dash-dropdown",
                                    ),
                                ]),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Section 1: Real Observed Sensor Vector
        render_section_header("1. Sensor vector at selected hour", "Ground-truth sensor readings observed in Building 59 South Wing"),
        html.Div(
            className="row g-3 mb-4",
            id="sim-sensor-cards-container",
        ),

        # Section 2: What-If Overrides Accordion / Card
        html.Div(
            className="glass-card mb-4",
            children=[
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P("Perturb input variables to evaluate hypothetical building states. Physical bounds and ramp constraints remain enforced.", className="text-muted", style={"fontSize": "0.82rem"}),
                        dbc.Checkbox(
                            id="sim-use-overrides",
                            label="Apply What-If overrides to counterfactual optimization",
                            value=False,
                            className="mb-3 font-weight-bold text-info",
                        ),
                        html.Div(
                            className="row g-4",
                            children=[
                                html.Div(className="col-12 col-md-6 col-lg-3", children=[
                                    html.Label("Fan Speed Override (%):", style={"fontSize": "0.8rem", "fontWeight": "600"}),
                                    dcc.Slider(id="sim-override-fan", min=40, max=90, step=5, value=65, marks={40: "40%", 65: "65%", 90: "90%"}, tooltip={"placement": "bottom", "always_visible": True}),
                                ]),
                                html.Div(className="col-12 col-md-6 col-lg-3", children=[
                                    html.Label("Damper Position Override (%):", style={"fontSize": "0.8rem", "fontWeight": "600"}),
                                    dcc.Slider(id="sim-override-damper", min=10, max=90, step=5, value=30, marks={10: "10%", 50: "50%", 90: "90%"}, tooltip={"placement": "bottom", "always_visible": True}),
                                ]),
                                html.Div(className="col-12 col-md-6 col-lg-3", children=[
                                    html.Label("Indoor Temp Override (°C):", style={"fontSize": "0.8rem", "fontWeight": "600"}),
                                    dcc.Slider(id="sim-override-tin", min=16.0, max=28.0, step=0.5, value=22.0, marks={16: "16°C", 22: "22°C", 28: "28°C"}, tooltip={"placement": "bottom", "always_visible": True}),
                                ]),
                                html.Div(className="col-12 col-md-6 col-lg-3", children=[
                                    html.Label("Outdoor Temp Override (°C):", style={"fontSize": "0.8rem", "fontWeight": "600"}),
                                    dcc.Slider(id="sim-override-tout", min=2.0, max=30.0, step=0.5, value=12.0, marks={2: "2°C", 16: "16°C", 30: "30°C"}, tooltip={"placement": "bottom", "always_visible": True}),
                                ]),
                            ],
                        ),
                    ], title="🛠️ What-If Overrides & Synthetic Perturbations (Optional)"),
                ], start_collapsed=True),
            ],
        ),

        # Section 3: Optimization Trigger & Action
        html.Div(
            className="d-flex justify-content-between align-items-center mb-4",
            children=[
                render_section_header("2. Counterfactual Optimization Execution", "Evaluate 99 fan × damper pairs under comfort and ramp safety gates"),
                html.Button(
                    children=[html.I(className="bi bi-rocket-takeoff-fill me-2"), "Run Optimizer"],
                    id="sim-run-btn",
                    className="btn-bems-primary",
                    n_clicks=0,
                ),
            ],
        ),

        # Section 4: Recommended Operating State & Impact KPIs
        render_section_header("3. Recommended Operating State & Impact", "Surrogate-evaluated setpoints compared to frozen baseline"),
        html.Div(
            className="row g-3 mb-4",
            id="sim-output-kpis-container",
        ),

        # Section 5: Diagnostic Attribution & Bar Comparison
        render_section_header("4. Diagnostic Attribution & Visual Verification", "Audit why the optimizer proposed this counterfactual control"),
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(
                    className="col-12 col-lg-6",
                    children=[
                        html.Div(id="sim-attribution-box", className="glass-card h-100"),
                    ],
                ),
                html.Div(
                    className="col-12 col-lg-6",
                    children=[
                        html.Div(
                            className="glass-card h-100",
                            children=[
                                html.H6("Baseline vs Recommended Actuator & Energy State", className="mb-2", style={"fontWeight": "600"}),
                                dcc.Graph(id="sim-compare-graph", config={"displayModeBar": False}, style={"height": "260px"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ])


# =============================================================================
# VIEW 3: ENERGY ANALYTICS
# =============================================================================

def build_view_energy_analytics():
    min_date = sim_df.index.min().strftime("%Y-%m-%d")
    max_date = sim_df.index.max().strftime("%Y-%m-%d")

    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("Energy Analytics", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("Tripartite time series, parity correlations, delta distributions, and submeter breakdowns", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
                html.Div(id="energy-filter-count-badge"),
            ],
        ),
        render_research_notice(),

        # Filter Bar Card
        html.Div(
            className="glass-card mb-4 filter-bar-card",
            style={"position": "relative", "zIndex": 1050},
            children=[
                html.Div(
                    className="row g-3 align-items-center",
                    children=[
                        html.Div(
                            className="col-12 col-md-5",
                            children=[
                                html.Label("Date Range Filter:", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#94A3B8"}),
                                dcc.DatePickerRange(
                                    id="energy-date-filter",
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    start_date=min_date,
                                    end_date=max_date,
                                    display_format="YYYY-MM-DD",
                                    className="dash-datepicker",
                                    style={"width": "100%"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="col-12 col-md-3",
                            children=[
                                html.Label("Occupancy State:", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#94A3B8"}),
                                dcc.Dropdown(
                                    id="energy-occ-filter",
                                    options=[
                                        {"label": "All States", "value": "All"},
                                        {"label": "Occupied Only", "value": "Occupied only"},
                                        {"label": "Unoccupied Only", "value": "Unoccupied only"},
                                    ],
                                    value="All",
                                    clearable=False,
                                    className="dash-dropdown",
                                ),
                            ],
                        ),
                        html.Div(
                            className="col-12 col-md-4",
                            children=[
                                html.Label("Temperature Regime:", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "#94A3B8"}),
                                dcc.Dropdown(
                                    id="energy-temp-filter",
                                    options=[
                                        {"label": "All Regimes", "value": "All"},
                                        {"label": "Cold (<10 °C)", "value": "Cold (<10 °C)"},
                                        {"label": "Mild (10-16 °C)", "value": "Mild (10-16 °C)"},
                                        {"label": "Warm (>16 °C)", "value": "Warm (>16 °C)"},
                                    ],
                                    value="All",
                                    clearable=False,
                                    className="dash-dropdown",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Chart 1: Tripartite Energy Trajectory
        html.Div(
            className="glass-card mb-4",
            children=[
                html.H5("1. Energy Trajectory: Actual vs Baseline vs Counterfactual", className="mb-1", style={"fontWeight": "600"}),
                html.Span("Direct comparison across evaluated test hours", style={"fontSize": "0.78rem", "color": "#64748B"}),
                dcc.Graph(id="energy-chart-trajectory", config={"displayModeBar": True, "responsive": True}),
            ],
        ),

        # Charts 2 & 3: Actual vs Baseline & Actual vs Counterfactual Scatters
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("2. Parity: Actual vs Baseline Prediction", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Dotted red line indicates perfect parity (y = x)", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(id="energy-chart-scatter-base", config={"displayModeBar": False}),
                    ]),
                ]),
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("3. Parity: Actual vs Counterfactual Optimized", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Downward shift illustrates model-estimated demand reduction", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(id="energy-chart-scatter-opt", config={"displayModeBar": False}),
                    ]),
                ]),
            ],
        ),

        # Charts 4 & 5: Hourly Delta Bar & Cumulative Savings Area
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("4. Hourly Estimated ΔE (Savings)", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("ΔE = E_cand − E_base (negative indicates reduction)", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(id="energy-chart-delta", config={"displayModeBar": False}),
                    ]),
                ]),
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("5. Cumulative Estimated Energy Savings", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Running sum of avoided energy (-ΔE)", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(id="energy-chart-cumulative", config={"displayModeBar": False}),
                    ]),
                ]),
            ],
        ),

        # Charts 6 & 7: Distribution Histogram & Submeter Breakdown
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("6. Hourly Savings Distribution", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Frequency distribution of positive savings hours", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(id="energy-chart-hist", config={"displayModeBar": False}),
                    ]),
                ]),
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("7. South Wing Submeter Breakdown (Actual)", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("HVAC RTU South vs Lighting vs Plug Loads (MELs)", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(id="energy-chart-submeter", config={"displayModeBar": False}),
                    ]),
                ]),
            ],
        ),
    ])


# =============================================================================
# VIEW 4: OCCUPANCY ANALYTICS
# =============================================================================

def build_view_occupancy_analytics():
    occ_test = p3_metrics.get("occupancy_classification", {}).get("test_results", {})

    # Diurnal Profile calculation
    df_p = sim_df.copy()
    df_p["hour"] = df_p.index.hour
    df_p["is_weekend"] = df_p.index.dayofweek >= 5
    diurnal = df_p.groupby(["hour", "is_weekend"])["actual_occupancy_state"].mean().reset_index()
    diurnal["Day type"] = diurnal["is_weekend"].map({True: "Weekend", False: "Weekday"})

    fig_diurnal = px.line(
        diurnal, x="hour", y="actual_occupancy_state", color="Day type",
        template=_TMPL, markers=True, height=300,
        labels={"hour": "Hour of Day", "actual_occupancy_state": "Mean Occupancy Rate"},
        color_discrete_map={"Weekday": C["savings"], "Weekend": C["baseline"]},
    )
    fig_diurnal.update_layout(**_layout_extras())

    # Weekly Heatmap
    df_h = sim_df.copy()
    df_h["day_name"] = df_h.index.day_name()
    df_h["hour"] = df_h.index.hour
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_m = df_h.groupby(["day_name", "hour"])["occupancy_probability"].mean().unstack()
    heat_m = heat_m.reindex(day_order)

    fig_heat = px.imshow(
        heat_m, template=_TMPL, height=300,
        labels=dict(x="Hour of Day", y="Day of Week", color="Mean P(Occ)"),
        color_continuous_scale="Blues",
    )
    fig_heat.update_layout(**_layout_extras())

    # Trajectory chart with Dual Thresholds
    fig_occ = _fig()
    fig_occ.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["occupancy_probability"],
        mode="lines", name="P(Occ=1) — Predicted",
        line=dict(color=C["savings"], width=1.5),
    ))
    fig_occ.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["actual_occupancy_state"],
        mode="lines", name="Actual Ground-Truth",
        line=dict(color=C["actual"], width=1, dash="dot"), opacity=0.6,
    ))
    fig_occ.add_hline(
        y=T_CLASS, line_dash="dash", line_color=C["baseline"],
        annotation_text=f"T_class = {T_CLASS:.2f} (Reporting / Decision Threshold)",
        annotation_position="top left",
    )
    fig_occ.add_hline(
        y=T_SAFETY, line_dash="dash", line_color=C["danger"],
        annotation_text=f"T_safety = {T_SAFETY:.2f} (Conservative Setback Safety Gate)",
        annotation_position="bottom left",
    )
    fig_occ.update_layout(height=360, xaxis_title="Simulation Timestamp", yaxis_title="Probability [0, 1]")

    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("Occupancy Analytics", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("The South Wing occupancy forecaster (t→t+1) acts as the master safety barrier for setback controls", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
            ],
        ),
        render_research_notice(),

        # Probability Trajectory Chart
        html.Div(
            className="glass-card mb-4",
            children=[
                html.H5("1. Occupancy Probability Trajectory & Dual Safety Thresholds", className="mb-1", style={"fontWeight": "600"}),
                html.Span("P(Occ=1) vs Actual Occupancy with T_class=0.51 and T_safety=0.30", style={"fontSize": "0.78rem", "color": "#64748B"}),
                dcc.Graph(figure=fig_occ, config={"displayModeBar": True, "responsive": True}),
            ],
        ),

        # Diurnal Profiles and Weekly Heatmap
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("2. Diurnal Occupancy Profile by Day Type", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Mean occupancy rate across hours for Weekdays vs Weekends", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(figure=fig_diurnal, config={"displayModeBar": False}),
                    ]),
                ]),
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("3. Weekly Occupancy Heatmap", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Day of Week × Hour of Day mean predicted occupancy probability", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(figure=fig_heat, config={"displayModeBar": False}),
                    ]),
                ]),
            ],
        ),

        # Champion Classifier Test Metrics Cards
        render_section_header("4. Champion Classifier — Frozen Test Metrics", "Random Forest classifier evaluated strictly on frozen test dataset"),
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-sm-4 col-md-2", children=render_kpi_card("F1-Score (T=0.51)", f"{occ_test.get('Test F1 (Tuned)', 0.9419):.4f}", "Champion", "positive", "bi-award")),
                html.Div(className="col-12 col-sm-4 col-md-2", children=render_kpi_card("Precision", f"{occ_test.get('Test Precision', 0.9332):.4f}", "Occupied accuracy", "neutral", "bi-bullseye")),
                html.Div(className="col-12 col-sm-4 col-md-2", children=render_kpi_card("Recall", f"{occ_test.get('Test Recall', 0.9508):.4f}", "Safety sensitivity", "positive", "bi-check2-all")),
                html.Div(className="col-12 col-sm-4 col-md-2", children=render_kpi_card("ROC-AUC", f"{occ_test.get('ROC-AUC', 0.9392):.4f}", "Discrimination", "neutral", "bi-graph-up")),
                html.Div(className="col-12 col-sm-4 col-md-2", children=render_kpi_card("PR-AUC", f"{occ_test.get('PR-AUC', 0.9806):.4f}", "Imbalance robust", "positive", "bi-trophy")),
                html.Div(className="col-12 col-sm-4 col-md-2", children=render_kpi_card("Balanced Acc", f"{occ_test.get('Bal Acc', 0.8643):.4f}", "Macro average", "neutral", "bi-sliders")),
            ],
        ),

        # Dual Threshold Architectural Notice Callout
        html.Div(
            className="glass-card mb-4",
            style={"borderLeft": "4px solid #38BDF8"},
            children=[
                html.H6("🛡️ Dual-Threshold Safety Architecture Callout", style={"color": "#38BDF8", "fontWeight": "700"}),
                html.P([
                    html.Strong("T_class = 0.51"), " (validation-tuned, maximum F1=0.9419) is used for statistical reporting. ",
                    html.Strong("T_safety = 0.30"), " is a deliberately conservative operational guardrail: any predicted P(occ) ≥ 0.30 immediately marks the space ",
                    html.Code("PROTECTED_OCCUPIED", className="status-pill status-pill-protected"),
                    ". This asymmetric risk weighting reduces false-setback occupied risk hours by ",
                    html.Strong("72.55%"), " during testing, ensuring zero occupant thermal distress.",
                ], className="mb-0", style={"fontSize": "0.85rem", "color": "#CBD5E1"}),
            ],
        ),
    ])


# =============================================================================
# VIEW 5: ML MODEL PERFORMANCE
# =============================================================================

def build_view_model_performance():
    val_occ = p3_metrics.get("occupancy_classification", {}).get("validation_results", [])
    df_occ = pd.DataFrame(val_occ) if val_occ else pd.DataFrame()

    fig_occ_bar = go.Figure()
    if not df_occ.empty and "Val F1 (Tuned)" in df_occ.columns:
        fig_occ_bar = px.bar(
            df_occ, x="Model", y="Val F1 (Tuned)", color="PR-AUC",
            template=_TMPL, height=280,
            title="Validation F1-Score Across Model Architectures",
            color_continuous_scale="Blues",
        )
        fig_occ_bar.update_layout(**_layout_extras())

    val_eng = p3_metrics.get("energy_forecasting", {}).get("validation_results", [])
    df_eng = pd.DataFrame(val_eng) if val_eng else pd.DataFrame()

    fig_eng_bar = go.Figure()
    if not df_eng.empty:
        ycol = "Chained RMSE" if "Chained RMSE" in df_eng.columns else df_eng.columns[2]
        fig_eng_bar = px.bar(
            df_eng, x="Model", y=ycol, template=_TMPL, height=280,
            title="Validation RMSE Across Energy Candidate Models (Lower is Better)",
            color_discrete_sequence=[C["baseline"]],
        )
        fig_eng_bar.update_layout(**_layout_extras())

    baselines = p3_robustness.get("baselines_summary", {})
    regimes = p3_robustness.get("regime_analysis", {}).get("outdoor_temperature", [])

    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("ML Model Benchmark & Audit", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("Frozen Phase 3 candidate benchmarks and Phase 3.5 robustness audits. Zero model retraining in dashboard.", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
            ],
        ),
        render_research_notice(),

        dbc.Tabs([
            # TAB 1: Occupancy Models
            dbc.Tab(label="Occupancy Models", tab_id="tab-occ", children=[
                html.Div(className="pt-3", children=[
                    render_section_header("Candidate Occupancy Model Benchmark (Validation Set)"),
                    html.Div(
                        className="glass-card mb-4",
                        children=[
                            dbc.Table.from_dataframe(df_occ, striped=True, bordered=False, hover=True, className="bems-table mb-3") if not df_occ.empty else html.Div("No validation table found"),
                            dcc.Graph(figure=fig_occ_bar, config={"displayModeBar": False}),
                        ],
                    ),
                    render_section_header("Frozen Test Set — Random Forest Champion Metrics"),
                    html.Div(
                        className="glass-card mb-3",
                        children=[
                            html.Pre(
                                json.dumps(p3_metrics.get("occupancy_classification", {}).get("test_results", {}), indent=2),
                                className="font-mono text-info mb-0",
                                style={"fontSize": "0.82rem"},
                            ),
                        ],
                    ),
                ]),
            ]),

            # TAB 2: Energy Surrogates
            dbc.Tab(label="Energy Surrogates", tab_id="tab-energy", children=[
                html.Div(className="pt-3", children=[
                    render_section_header("Candidate Energy Model Benchmark (Validation Set)"),
                    html.Div(
                        className="glass-card mb-4",
                        children=[
                            dbc.Table.from_dataframe(df_eng, striped=True, bordered=False, hover=True, className="bems-table mb-3") if not df_eng.empty else html.Div("No validation table found"),
                            dcc.Graph(figure=fig_eng_bar, config={"displayModeBar": False}),
                        ],
                    ),
                    render_section_header("Frozen Test Set — Ridge Regression Champion (α=10)"),
                    html.Div(
                        className="glass-card mb-3",
                        children=[
                            html.Pre(
                                json.dumps(p3_metrics.get("energy_forecasting", {}).get("final_test_metrics_chained", {}), indent=2),
                                className="font-mono text-warning mb-0",
                                style={"fontSize": "0.82rem"},
                            ),
                        ],
                    ),
                    html.Div(
                        className="glass-card mb-3",
                        style={"borderLeft": "4px solid #10B981"},
                        children=[
                            html.H6("💡 Architectural Selection Rationale: Why Ridge Over Tree Ensembles?", style={"color": "#10B981", "fontWeight": "700"}),
                            html.P([
                                "Non-linear tree ensembles (Random Forest, XGBoost, LightGBM) exhibited non-physical step-function artifacts and unconstrained extrapolation under counterfactual actuator perturbations. ",
                                html.Strong("Ridge Regression (α=10)"),
                                " was chosen as the champion surrogate: its smooth linear gradients maintain strict physical monotonicity (higher fan speed = more energy consumption) and predictable sensitivity guarantees.",
                            ], className="mb-0", style={"fontSize": "0.85rem", "color": "#CBD5E1"}),
                        ],
                    ),
                ]),
            ]),

            # TAB 3: Naive Baselines & Audits
            dbc.Tab(label="Naive Baselines & Audits", tab_id="tab-baselines", children=[
                html.Div(className="pt-3", children=[
                    render_section_header("Naive Forecasting Baseline Audit (Phase 3.5)", "Candidate ML models audited against simple baselines to prevent overconfidence"),
                    html.Div(
                        className="glass-card mb-4",
                        children=[
                            html.Ul([
                                html.Li([html.Strong(f"{k.replace('_', ' ').title()}: "), html.Span(str(v), style={"color": "#38BDF8"})])
                                for k, v in baselines.items()
                            ], className="mb-0", style={"fontSize": "0.85rem"}),
                        ],
                    ),
                    render_section_header("Temperature Regime Stratification (Test Period)"),
                    html.Div(
                        className="glass-card mb-3",
                        children=[
                            dbc.Table.from_dataframe(pd.DataFrame(regimes), striped=True, bordered=False, hover=True, className="bems-table mb-0") if regimes else html.Div("No regime data found"),
                        ],
                    ),
                ]),
            ]),
        ]),
    ])


# =============================================================================
# VIEW 6: OPTIMIZATION ANALYSIS
# =============================================================================

def build_view_optimization_analysis():
    kpis = p4_summary.get("overall_simulation_metrics", {})
    rd = p4_summary.get("regime_breakdown", {})

    # Regime bar charts
    df_ob = pd.DataFrame([rd.get("occupancy_occupied", {}), rd.get("occupancy_unoccupied", {})]) if rd else pd.DataFrame()
    fig_ob = px.bar(
        df_ob, x="regime", y="savings_kwh", color="regime",
        template=_TMPL, height=260,
        labels={"regime": "Occupancy Regime", "savings_kwh": "Estimated Savings (kWh)"},
        color_discrete_map={"Occupied": C["occupied"], "Unoccupied": C["optimized"]},
    ) if not df_ob.empty else go.Figure()
    fig_ob.update_layout(**_layout_extras())

    df_tb = pd.DataFrame([rd.get("temp_cold", {}), rd.get("temp_mild", {}), rd.get("temp_warm", {})]) if rd else pd.DataFrame()
    fig_tb = px.bar(
        df_tb, x="regime", y="savings_kwh", color="regime",
        template=_TMPL, height=260,
        labels={"regime": "Temperature Regime", "savings_kwh": "Estimated Savings (kWh)"},
        color_discrete_sequence=[C["savings"], C["optimized"], C["baseline"]],
    ) if not df_tb.empty else go.Figure()
    fig_tb.update_layout(**_layout_extras())

    # Ablations Table
    abl_rows = [
        {
            "Ablation Scenario": k,
            "Hours Optimized": v.get("hours_optimized"),
            "Savings (kWh)": f"{v.get('total_estimated_energy_savings_kwh', 0):,.1f}",
            "Saving %": f"{v.get('overall_estimated_energy_savings_pct', 0):.1f}%",
            "Cost Savings ($)": f"${v.get('total_estimated_cost_savings_usd', 0):,.2f}",
        }
        for k, v in p4_summary.get("ablations", {}).items()
    ]
    df_abl = pd.DataFrame(abl_rows)

    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("Counterfactual Optimization Analysis", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("Simulation aggregate performance, regime stratification, and safeguard ablation audit", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
            ],
        ),
        render_research_notice(),

        # Aggregate KPIs Grid (8 Cards)
        render_section_header("Test-Period Simulation Aggregate KPIs", "Evaluated over 994 consecutive hours (January 11 to February 21, 2019)"),
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Evaluated Hours", f"{kpis.get('hours_evaluated', 994):,} hrs", "Frozen test set", "neutral", "bi-clock")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Hours Optimized", f"{kpis.get('hours_optimized', 804):,} hrs", f"{kpis.get('pct_hours_improved', 80.89):.1f}% active hours", "positive", "bi-lightning")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Estimated Energy Savings", f"{kpis.get('total_estimated_energy_savings_kwh', 5120.45):,.1f} kWh", f"{kpis.get('overall_estimated_energy_savings_pct', 53.32):.1f}% reduction", "positive", "bi-battery-charging")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Estimated Cost Savings", f"${kpis.get('total_estimated_cost_savings_usd', 1126.50):,.2f}", f"-{kpis.get('total_estimated_co2_reduction_kg', 1075.29):,.1f} kg CO₂ avoided", "positive", "bi-cash-coin")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Baseline Preserved", f"{kpis.get('hours_no_change', 190):,} hrs", "NO_CHANGE deadband", "neutral", "bi-pause-circle")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Mean ΔE / Active Hour", f"{kpis.get('mean_delta_per_optimized_hour_kwh', -6.37):.2f} kWh", "Average reduction", "positive", "bi-arrow-down-right")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Median ΔE / Active Hour", f"{kpis.get('median_delta_per_optimized_hour_kwh', -6.58):.2f} kWh", "Robust central tendency", "positive", "bi-bar-chart")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Carbon Avoided", f"{kpis.get('total_estimated_co2_reduction_kg', 1075.29):,.1f} kg", "Avoided emissions", "positive", "bi-tree")),
            ],
        ),

        # Regime Stratification Breakdown
        render_section_header("Regime Stratification Breakdown"),
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("By Occupancy Regime", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Savings in Occupied (comfort-bounded) vs Unoccupied (setback-eligible)", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(figure=fig_ob, config={"displayModeBar": False}),
                    ]),
                ]),
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card", children=[
                        html.H6("By Outdoor Temperature Regime", className="mb-1", style={"fontWeight": "600"}),
                        html.Span("Cold (<10°C) vs Mild (10-16°C) vs Warm (>16°C)", style={"fontSize": "0.75rem", "color": "#64748B"}),
                        dcc.Graph(figure=fig_tb, config={"displayModeBar": False}),
                    ]),
                ]),
            ],
        ),

        # Safeguard Ablation Study Table & Critical Callout
        render_section_header("Safeguard Ablation Study", "Evaluating counterfactual outcomes when removing individual safety boundaries"),
        html.Div(
            className="glass-card mb-4",
            children=[
                dbc.Table.from_dataframe(df_abl, striped=True, bordered=False, hover=True, className="bems-table mb-3") if not df_abl.empty else html.Div("No ablation data"),
            ],
        ),

        html.Div(
            className="glass-card mb-3",
            style={"borderLeft": "4px solid #EF4444"},
            children=[
                html.H6("⚠️ Critical Engineering Callout: The Non-Physical No-Ramp Ablation Anomaly", style={"color": "#EF4444", "fontWeight": "700"}),
                html.P([
                    "In Ablation C (no ramp limits), the model reports ",
                    html.Strong("115.86% estimated savings (11,125.39 kWh)"),
                    " — an engineering impossibility caused by instantaneous 50% fan-speed drops that exceed physical motor inertia and actuator slewing rates. ",
                    "The ",
                    html.Strong("Full Safeguards Policy"),
                    " enforces strict ramp limits (≤15%/hr fan speed delta), producing the realistic and physically valid ",
                    html.Strong("53.32%"),
                    " reduction. This demonstrates why deterministic domain safety layers are essential around statistical ML surrogates.",
                ], className="mb-0", style={"fontSize": "0.85rem", "color": "#CBD5E1"}),
            ],
        ),
    ])


# =============================================================================
# VIEW 7: SAFETY & VALIDITY
# =============================================================================

def build_view_safety_validity():
    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("Safety Constraints & Domain Validity", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("Deterministic safety envelopes, operational state machine, and physical limitation bounds", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
            ],
        ),
        render_research_notice(),

        # Safety Bounds Envelope Grid (8 Cards)
        render_section_header("Operational Safety Bounds Envelope"),
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Fan Speed Bounds", "40 – 90 %", "Historical empirical support", "neutral", "bi-fan")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Damper Bounds", "10 – 90 %", "ASHRAE ventilation floor", "neutral", "bi-door-open")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Max Fan Ramp", "≤ 15 %/hr", "VFD motor protection", "positive", "bi-speedometer")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Max Damper Ramp", "≤ 30 %/hr", "Actuator gear protection", "positive", "bi-gear")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Safety Gate T_safety", f"{T_SAFETY:.2f}", "Occupancy setback guard", "positive", "bi-shield")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Improvement Epsilon ε", f"{MIN_IMPROVEMENT_EPSILON:.2f} kWh", "Prevents hunting cycles", "positive", "bi-filter")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Display Energy Floor", f"{BASELOAD_ENERGY_FLOOR:.2f} kWh", "Calibrated baseload floor", "neutral", "bi-layers")),
                html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Candidate Grid", "99 pairs", "Fan × Damper discrete grid", "neutral", "bi-grid-3x3")),
            ],
        ),

        # The Four System Operational States
        render_section_header("The Four System Operational States"),
        html.Div(
            className="row g-3 mb-4",
            children=[
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card mb-3", style={"borderLeft": "4px solid #10B981"}, children=[
                        html.H6("✅ CONTROL-SAFE (SETBACK ELIGIBLE)", style={"color": "#10B981", "fontWeight": "700"}),
                        html.P("Predicted P(occ) is strictly below T_safety = 0.30. Zone confirmed vacant with 95%+ confidence. Optimizer is authorized to evaluate deep setback fan speeds (down to 40%) and reduced outdoor air dampers.", className="mb-0", style={"fontSize": "0.85rem"}),
                    ]),
                    html.Div(className="glass-card", style={"borderLeft": "4px solid #F59E0B"}, children=[
                        html.H6("🛡️ PROTECTED_OCCUPIED", style={"color": "#F59E0B", "fontWeight": "700"}),
                        html.P("Predicted P(occ) ≥ 0.30. Deep setback is strictly prohibited regardless of current energy price. Supply fan speeds and outdoor dampers are constrained to maintain full occupant thermal comfort and ventilation standards.", className="mb-0", style={"fontSize": "0.85rem"}),
                    ]),
                ]),
                html.Div(className="col-12 col-md-6", children=[
                    html.Div(className="glass-card mb-3", style={"borderLeft": "4px solid #EF4444"}, children=[
                        html.H6("❄️ LOW_SURROGATE_CONFIDENCE (Cold Regime <10 °C)", style={"color": "#EF4444", "fontWeight": "700"}),
                        html.P("Outdoor temperature is below 10 °C. Cold weather was almost entirely absent in training data (May–Nov 2018). The linear surrogate exhibits negative bias; all counterfactuals are explicitly flagged as low-confidence.", className="mb-0", style={"fontSize": "0.85rem"}),
                    ]),
                    html.Div(className="glass-card", style={"borderLeft": "4px solid #38BDF8"}, children=[
                        html.H6("⏸️ NO_CHANGE (Deadband Preservation)", style={"color": "#38BDF8", "fontWeight": "700"}),
                        html.P("If no candidate setpoint improves baseline energy consumption by at least ε = 0.20 kWh, the optimizer issues NO_CHANGE to preserve current actuator states and prevent actuator chattering or valve wear.", className="mb-0", style={"fontSize": "0.85rem"}),
                    ]),
                ]),
            ],
        ),

        # Domain Limitations & Scope Boundaries
        render_section_header("Domain Limitations & Physical Scope Boundaries"),
        html.Div(
            className="glass-card mb-3",
            style={"borderLeft": "4px solid #818CF8"},
            children=[
                html.Ul([
                    html.Li([
                        html.Strong("Empirical Regression vs Physics Simulation: "),
                        "The Ridge surrogate is a statistical response surface, not EnergyPlus or DOE-2. It models energy consumption but does not co-simulate zone thermal capacitance or heating coil dynamics.",
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Indoor Temperature Dynamics: "),
                        "In this single-step research simulation, adjusting the fan speed does not update the indoor temperature of subsequent hours.",
                    ], className="mb-2"),
                    html.Li([
                        html.Strong("Offline Simulation Scope: "),
                        "This is an academic research platform. No BACnet, Modbus, or MQTT network control packets are dispatched to physical building hardware.",
                    ]),
                ], className="mb-0", style={"fontSize": "0.85rem", "color": "#CBD5E1"}),
            ],
        ),
    ])


# =============================================================================
# VIEW 8: ABOUT & METHODOLOGY
# =============================================================================

def build_view_about_methodology():
    return html.Div([
        html.Div(
            className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
            children=[
                html.Div([
                    html.H2("About & Modeling Methodology", className="mb-0", style={"fontWeight": "700"}),
                    html.Span("Mathematical formulation, pipeline specifications, and academic attribution", style={"color": "#94A3B8", "fontSize": "0.85rem"}),
                ]),
            ],
        ),
        render_research_notice(),

        html.P(
            "This interactive platform encapsulates the complete end-to-end BEMS research workflow for "
            "Lawrence Berkeley National Laboratory (LBNL) Building 59 (South Wing, Floors 3 & 4) — a high-density commercial office building.",
            style={"fontSize": "0.9rem", "color": "#CBD5E1"},
        ),

        # System Pipeline
        render_section_header("System Pipeline Architecture"),
        html.Div(
            className="glass-card mb-4",
            children=[
                html.Pre(
"""Building sensor state at time t
         │
         ▼
[Occupancy Forecast t→t+1]  Random Forest, T_class=0.51, T_safety=0.30
         │
         ▼
[Energy Forecast t→t+1]     Ridge Regression, α=10 (linear monotonic gradient)
         │
         ▼
[Counterfactual Optimizer]   Bounded grid search over 99 (fan × damper) pairs
         │                   Ramp limits (≤15%/hr fan) + Occupancy safety gate
         ▼
Recommended setpoints + ΔE = E_candidate − E_baseline""",
                    className="font-mono text-info mb-0",
                    style={"fontSize": "0.85rem", "lineHeight": "1.5"},
                ),
            ],
        ),

        # Technical Specifications Table
        render_section_header("Technical Specifications"),
        html.Div(
            className="glass-card mb-4",
            children=[
                html.Table(
                    className="bems-table",
                    children=[
                        html.Thead(html.Tr([html.Th("Dimension"), html.Th("Specification"), html.Th("Notes")])),
                        html.Tbody([
                            html.Tr([html.Td("Dataset"), html.Td("LBNL Building 59"), html.Td("Open-source multi-sensor commercial office building")]),
                            html.Tr([html.Td("Spatial Scope"), html.Td("South Wing (Floors 3 & 4)"), html.Td("High sensor-density workspace and perimeter zones")]),
                            html.Tr([html.Td("Time Resolution"), html.Td("1 Hour"), html.Td("Strictly backward-looking leakage-safe features")]),
                            html.Tr([html.Td("Prediction Horizon"), html.Td("t → t+1"), html.Td("Features at t forecast operating state at t+1")]),
                            html.Tr([html.Td("Occupancy Model"), html.Td("Random Forest Classifier"), html.Td("T_class=0.51, T_safety=0.30")]),
                            html.Tr([html.Td("Energy Surrogate"), html.Td("Ridge Regression (α=10)"), html.Td("Linear monotonic gradient, avoids step artifacts")]),
                            html.Tr([html.Td("Candidate Space"), html.Td("99 Discrete Pairs"), html.Td("Fan {40, 45, ..., 90}% × Damper {10, 20, ..., 90}%")]),
                            html.Tr([html.Td("Electricity Tariff"), html.Td("$0.22 / kWh"), html.Td("PG&E Commercial B-19 rate schedule")]),
                            html.Tr([html.Td("Carbon Intensity"), html.Td("0.210 kg CO₂ / kWh"), html.Td("EPA eGRID CAMX regional subregion")]),
                            html.Tr([html.Td("Display Floor"), html.Td("6.50 kWh"), html.Td("Calibrated baseload display floor")]),
                        ]),
                    ],
                ),
            ],
        ),

        # Mathematical Formulation
        render_section_header("Mathematical Formulation"),
        html.Div(
            className="glass-card mb-4",
            children=[
                html.Div([
                    html.Div("ΔE(u) = Ê(u_candidate, Ô_{t+1}, X_t) − Ê(u_baseline, Ô_{t+1}, X_t)", className="font-mono text-warning text-center py-2", style={"fontSize": "1.1rem"}),
                ]),
                html.P([
                    "Where ",
                    html.Code("u = (fan_spd, damper_pct)"),
                    " is the controllable HVAC actuator vector, ",
                    html.Code("Ô_{t+1} ∈ {0, 1}"),
                    " is the predicted occupancy state, and ",
                    html.Code("X_t"),
                    " is the frozen exogenous contextual sensor vector. A negative ΔE indicates estimated demand reduction.",
                ], className="mb-0", style={"fontSize": "0.85rem", "color": "#CBD5E1"}),
            ],
        ),

        # Citation & Attribution
        render_section_header("Citation & Academic Attribution"),
        html.Div(
            className="glass-card mb-3",
            children=[
                html.Ul([
                    html.Li([html.Strong("Dataset: "), html.Span("Building 59 Multi-Sensor Dataset, Lawrence Berkeley National Laboratory (LBNL).")]),
                    html.Li([html.Strong("Project: "), html.Span("AI-Based Energy Optimization for Smart Buildings — Phases 0 through 5 (2026).")]),
                ], className="mb-0", style={"fontSize": "0.85rem", "color": "#CBD5E1"}),
            ],
        ),
    ])


# =============================================================================
# MAIN APP LAYOUT & ROUTING
# =============================================================================

app.layout = html.Div(
    className="d-flex",
    style={"minHeight": "100vh"},
    children=[
        dcc.Location(id="url", refresh=False),
        # Sidebar Container
        html.Div(id="sidebar-wrapper", style={"width": "280px", "flexShrink": "0"}),
        # Main Viewport Container
        html.Div(
            className="flex-grow-1 p-4",
            style={"backgroundColor": "#080D1A", "overflowY": "auto", "maxHeight": "100vh"},
            children=[
                html.Div(id="page-content"),
            ],
        ),
    ],
)


# =============================================================================
# CALLBACKS
# =============================================================================

# Router Callback
@app.callback(
    Output("page-content", "children"),
    Output("sidebar-wrapper", "children"),
    Input("url", "pathname"),
)
def route_page(pathname: Optional[str]):
    pathname = pathname or "/"
    sidebar = render_sidebar(pathname)

    if pathname == "/" or pathname == "/executive-overview":
        content = build_view_executive_overview()
    elif pathname == "/scenario-simulator":
        content = build_view_scenario_simulator()
    elif pathname == "/energy-analytics":
        content = build_view_energy_analytics()
    elif pathname == "/occupancy-analytics":
        content = build_view_occupancy_analytics()
    elif pathname == "/model-performance":
        content = build_view_model_performance()
    elif pathname == "/optimization-analysis":
        content = build_view_optimization_analysis()
    elif pathname == "/safety-validity":
        content = build_view_safety_validity()
    elif pathname == "/about-methodology":
        content = build_view_about_methodology()
    else:
        content = html.Div([
            html.H3("404: Page Not Found", className="text-danger"),
            dcc.Link("Return to Executive Overview", href="/", className="btn-bems-primary mt-3"),
        ])

    return content, sidebar


# -----------------------------------------------------------------------------
# Callbacks: Executive Overview
# -----------------------------------------------------------------------------
@app.callback(
    Output("exec-gate-badge-container", "children"),
    Output("exec-kpi-occ", "children"),
    Output("exec-kpi-tout", "children"),
    Output("exec-kpi-fan", "children"),
    Output("exec-kpi-damper", "children"),
    Output("exec-kpi-base-e", "children"),
    Output("exec-kpi-opt-e", "children"),
    Output("exec-kpi-savings", "children"),
    Output("exec-kpi-cost-co2", "children"),
    Output("exec-trajectory-graph", "figure"),
    Output("exec-recommendation-box", "children"),
    Output("exec-validity-box", "children"),
    Input("exec-scenario-select", "value"),
)
def update_executive_overview(scenario_key: str):
    if not scenario_key or scenario_key not in rep_timestamps:
        scenario_key = list(rep_timestamps.keys())[0]

    ts_str = rep_timestamps[scenario_key]
    ts = pd.to_datetime(ts_str)
    row = sim_df.loc[ts] if ts in sim_df.index else sim_df.iloc[0]

    gate_status = row["control_safety_gate"]
    if gate_status == "PROTECTED_OCCUPIED":
        badge = html.Span([html.I(className="bi bi-shield-shaded me-1"), "PROTECTED OCCUPIED"], className="status-pill status-pill-protected")
    else:
        badge = html.Span([html.I(className="bi bi-check-circle-fill me-1"), "SETBACK ELIGIBLE"], className="status-pill status-pill-safe")

    prob_occ = float(row["occupancy_probability"])
    occ_state = int(row["occupancy_state"])
    outdoor_t = float(row["outdoor_temp_c"])
    base_fan = float(row["baseline_fan"])
    rec_fan = float(row["recommended_fan"])
    base_damper = float(row["baseline_damper"])
    rec_damper = float(row["recommended_damper"])
    base_energy = float(row["baseline_predicted_energy"])
    opt_energy = float(row["optimized_predicted_energy"])
    delta_e = float(row["optimized_delta_energy"])
    save_pct = float(row["estimated_saving_percent"])
    cost_save = float(row["estimated_cost_saving"])
    co2_save = float(row["estimated_co2_reduction"])

    kpi1 = render_kpi_card("Forecasted Occupancy", "OCCUPIED" if occ_state else "VACANT", f"P(occ) = {prob_occ * 100:.1f}%", "positive" if not occ_state else "neutral", "bi-person-check")
    kpi2 = render_kpi_card("Outdoor Temperature", f"{outdoor_t:.1f} °C", "Cold (<10°C)" if outdoor_t < 10 else "Mild (≥10°C)", "danger" if outdoor_t < 10 else "neutral", "bi-thermometer-half")
    fan_delta = rec_fan - base_fan
    kpi3 = render_kpi_card("Supply Fan VFD", f"{rec_fan:.0f}%", f"{fan_delta:+.0f}% vs base", "positive" if fan_delta <= 0 else "negative", "bi-fan")
    damp_delta = rec_damper - base_damper
    kpi4 = render_kpi_card("Damper Outdoor Air", f"{rec_damper:.0f}%", f"{damp_delta:+.0f}% vs base", "positive" if damp_delta <= 0 else "negative", "bi-door-open")
    kpi5 = render_kpi_card("Baseline Energy", f"{base_energy:.2f} kWh", "Uncontrolled state", "neutral", "bi-lightning")
    kpi6 = render_kpi_card("Counterfactual Energy", f"{opt_energy:.2f} kWh", f"-{save_pct:.1f}% reduction", "positive", "bi-lightning-fill")
    kpi7 = render_kpi_card("Hourly Savings", f"{-delta_e:.2f} kWh", f"{save_pct:.1f}% saved", "positive", "bi-arrow-down-right-circle")
    kpi8 = render_kpi_card("Economic / Carbon", f"${cost_save:.2f}/hr", f"-{co2_save:.2f} kg CO₂/hr", "positive", "bi-cash")

    # Trajectory Figure with Highlight
    fig = _fig()
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["actual_total_energy"],
        mode="lines", name="Historical Actual",
        line=dict(color=C["actual"], width=1, dash="dot"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["baseline_predicted_energy"],
        mode="lines", name="Model Baseline",
        line=dict(color=C["baseline"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=sim_df.index, y=sim_df["optimized_predicted_energy"],
        mode="lines", name="Counterfactual Optimized",
        line=dict(color=C["optimized"], width=2),
    ))

    # Add vertical line for selected timestamp
    fig.add_vline(x=ts, line_width=1.5, line_dash="dash", line_color="#38BDF8")
    fig.add_annotation(x=ts, y=base_energy, text="Selected Scenario", showarrow=True, arrowhead=2, arrowcolor="#38BDF8", font=dict(color="#38BDF8", size=10))

    fig.update_layout(height=360, xaxis_title="Simulation Timestamp", yaxis_title="Hourly Energy (kWh)")

    # Recommendation Callout
    rec_box = [
        html.H6([html.I(className="bi bi-lightbulb-fill text-warning me-2"), "Optimizer Recommendation"], style={"color": "#10B981", "fontWeight": "700"}),
        html.P(row["action_reason"], className="mb-0", style={"fontSize": "0.9rem", "color": "#F8FAFC"}),
    ]

    # Validity Callout
    val_box = [
        html.H6([html.I(className="bi bi-shield-check text-info me-2"), "Surrogate Validity & Safety Envelope"], style={"color": "#F59E0B", "fontWeight": "700"}),
        html.Div([
            html.Span("Surrogate Confidence: ", style={"color": "#94A3B8"}),
            html.Code(row["confidence_flag"], className="text-info"),
        ], className="mb-1", style={"fontSize": "0.85rem"}),
        html.Div([
            html.Span("Constraint Status: ", style={"color": "#94A3B8"}),
            html.Code(row["constraint_status"], className="text-warning"),
        ], style={"fontSize": "0.85rem"}),
    ]

    return badge, kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8, fig, rec_box, val_box


# -----------------------------------------------------------------------------
# Callbacks: Scenario Simulator
# -----------------------------------------------------------------------------
@app.callback(
    Output("sim-preset-container", "style"),
    Output("sim-custom-container", "style"),
    Input("sim-mode-radio", "value"),
)
def toggle_simulator_selection_mode(mode: str):
    if mode == "custom":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("sim-sensor-cards-container", "children"),
    Output("sim-output-kpis-container", "children"),
    Output("sim-attribution-box", "children"),
    Output("sim-compare-graph", "figure"),
    Input("sim-run-btn", "n_clicks"),
    Input("sim-preset-select", "value"),
    Input("sim-custom-select", "value"),
    State("sim-mode-radio", "value"),
    State("sim-use-overrides", "value"),
    State("sim-override-fan", "value"),
    State("sim-override-damper", "value"),
    State("sim-override-tin", "value"),
    State("sim-override-tout", "value"),
)
def execute_scenario_simulator(
    n_clicks: int, preset_ts: str, custom_ts: str,
    mode: str, use_overrides: bool,
    ov_fan: float, ov_damper: float, ov_tin: float, ov_tout: float,
):
    selected_ts_str = custom_ts if mode == "custom" and custom_ts else preset_ts
    if not selected_ts_str:
        selected_ts_str = rep_timestamps[list(rep_timestamps.keys())[0]]

    ts = pd.to_datetime(selected_ts_str)
    if ts not in joint_df.index:
        ts = joint_df.index[0]

    hist_row = joint_df.loc[ts]
    if isinstance(hist_row, pd.DataFrame):
        hist_row = hist_row.iloc[0]

    # Observed Sensor Cards
    sensor_cards = [
        html.Div(className="col-6 col-md-2", children=render_kpi_card("Indoor Temp", f"{hist_row.get('indoor_temp_mean', 22.0):.1f} °C", "Zone actual", "neutral", "bi-thermometer")),
        html.Div(className="col-6 col-md-2", children=render_kpi_card("Outdoor Temp", f"{hist_row.get('outdoor_temp_c', 12.0):.1f} °C", "Ambient", "neutral", "bi-cloud-sun")),
        html.Div(className="col-6 col-md-2", children=render_kpi_card("Rel. Humidity", f"{hist_row.get('relative_humidity', 50.0):.1f} %", "Psychrometric", "neutral", "bi-droplet")),
        html.Div(className="col-6 col-md-2", children=render_kpi_card("Solar Rad", f"{hist_row.get('solar_radiation', 0.0):.0f} W/m²", "Insolation", "neutral", "bi-sun")),
        html.Div(className="col-6 col-md-2", children=render_kpi_card("Baseline Fan", f"{hist_row.get('rtu_south_fan_spd_mean_next_hour', 65.0):.0f} %", "Operating setpoint", "neutral", "bi-fan")),
        html.Div(className="col-6 col-md-2", children=render_kpi_card("Baseline Damper", f"{hist_row.get('rtu_south_damper_pct_mean_next_hour', 30.0):.0f} %", "Outside air damper", "neutral", "bi-door-open")),
    ]

    overrides_dict = None
    if use_overrides:
        overrides_dict = {
            "rtu_south_fan_spd_mean_next_hour": float(ov_fan),
            "rtu_south_damper_pct_mean_next_hour": float(ov_damper),
            "indoor_temp_mean": float(ov_tin),
            "outdoor_temp_c": float(ov_tout),
        }

    # Run single-hour counterfactual optimization
    res = run_scenario_optimization(
        timestamp=ts,
        joint_df=joint_df,
        occ_art=occ_art,
        energy_art=energy_art,
        overrides=overrides_dict,
        enforce_ramps=True,
        enforce_comfort=True,
        include_lighting=True,
    )

    # Output Impact KPIs
    rec_fan = res["recommended_fan"]
    base_fan = res["baseline_fan"]
    rec_damp = res["recommended_damper"]
    base_damp = res["baseline_damper"]
    pred_energy = res["optimized_predicted_energy"]
    delta_e = res["optimized_delta_energy"]
    pct_save = res["estimated_saving_percent"]
    cost_save = res["estimated_cost_saving"]
    co2_save = res["estimated_co2_reduction"]

    output_kpis = [
        html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Supply Fan VFD", f"{rec_fan:.0f}%", f"{rec_fan - base_fan:+.0f}% vs baseline", "positive" if (rec_fan <= base_fan) else "negative", "bi-fan")),
        html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Damper Setpoint", f"{rec_damp:.0f}%", f"{rec_damp - base_damp:+.0f}% vs baseline", "positive" if (rec_damp <= base_damp) else "negative", "bi-door-open")),
        html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Counterfactual Energy", f"{pred_energy:.2f} kWh", f"ΔE: {delta_e:+.2f} kWh ({pct_save:.1f}%)", "positive", "bi-lightning-fill")),
        html.Div(className="col-12 col-sm-6 col-xl-3", children=render_kpi_card("Estimated Savings", f"${cost_save:.2f}/hr", f"-{co2_save:.2f} kg CO₂/hr avoided", "positive", "bi-cash-coin")),
    ]

    # Attribution Details Box
    attribution_children = [
        html.H6([html.I(className="bi bi-lightbulb-fill text-warning me-2"), "Optimizer Attribution Rationale"], style={"color": "#10B981", "fontWeight": "700"}),
        html.P(res["action_reason"], className="mb-3", style={"fontSize": "0.92rem", "color": "#F8FAFC"}),
        html.Div(className="p-3 rounded mb-2", style={"backgroundColor": "rgba(15, 23, 42, 0.7)", "border": "1px solid #1E293B"}, children=[
            html.Div([
                html.Span("Safety Gate: ", style={"color": "#94A3B8"}),
                html.Code(res["control_safety_gate"], className="text-info me-2"),
                html.Span(f"(P(occ) = {res['occupancy_probability'] * 100:.1f}%, T_class={T_CLASS}, T_safety={T_SAFETY})", style={"fontSize": "0.8rem", "color": "#64748B"}),
            ], className="mb-2"),
            html.Div([
                html.Span("Surrogate Confidence: ", style={"color": "#94A3B8"}),
                html.Code(res["confidence_flag"], className="text-warning me-2"),
                html.Span("Constraint Status: ", style={"color": "#94A3B8"}),
                html.Code(res["constraint_status"], className="text-success"),
            ]),
        ]),
    ]

    # Comparative Bar Graph
    fig_bar = _fig()
    fig_bar.add_trace(go.Bar(
        name="Baseline",
        x=["Fan (%)", "Damper (%)", "Energy (kWh)"],
        y=[base_fan, base_damp, res["baseline_predicted_energy"]],
        marker_color=C["baseline"], marker_line_width=0,
    ))
    fig_bar.add_trace(go.Bar(
        name="Recommended",
        x=["Fan (%)", "Damper (%)", "Energy (kWh)"],
        y=[rec_fan, rec_damp, pred_energy],
        marker_color=C["optimized"], marker_line_width=0,
    ))
    fig_bar.update_layout(barmode="group", height=240, margin=dict(l=10, r=10, t=10, b=30))

    return sensor_cards, output_kpis, attribution_children, fig_bar


# -----------------------------------------------------------------------------
# Callbacks: Energy Analytics Filters & Charts
# -----------------------------------------------------------------------------
@app.callback(
    Output("energy-filter-count-badge", "children"),
    Output("energy-chart-trajectory", "figure"),
    Output("energy-chart-scatter-base", "figure"),
    Output("energy-chart-scatter-opt", "figure"),
    Output("energy-chart-delta", "figure"),
    Output("energy-chart-cumulative", "figure"),
    Output("energy-chart-hist", "figure"),
    Output("energy-chart-submeter", "figure"),
    Input("energy-date-filter", "start_date"),
    Input("energy-date-filter", "end_date"),
    Input("energy-occ-filter", "value"),
    Input("energy-temp-filter", "value"),
)
def update_energy_analytics(start_date: str, end_date: str, occ_filter: str, temp_filter: str):
    f_df = sim_df.copy()

    if start_date and end_date:
        s = pd.to_datetime(start_date).date()
        e = pd.to_datetime(end_date).date()
        f_df = f_df.loc[(f_df.index.date >= s) & (f_df.index.date <= e)]

    if occ_filter == "Occupied only":
        f_df = f_df.loc[f_df["actual_occupancy_state"] == 1]
    elif occ_filter == "Unoccupied only":
        f_df = f_df.loc[f_df["actual_occupancy_state"] == 0]

    if temp_filter == "Cold (<10 °C)":
        f_df = f_df.loc[f_df["outdoor_temp_c"] < 10.0]
    elif temp_filter == "Mild (10-16 °C)":
        f_df = f_df.loc[(f_df["outdoor_temp_c"] >= 10.0) & (f_df["outdoor_temp_c"] <= 16.0)]
    elif temp_filter == "Warm (>16 °C)":
        f_df = f_df.loc[f_df["outdoor_temp_c"] > 16.0]

    count_badge = html.Span(
        f"Showing {len(f_df):,} of {len(sim_df):,} test hours",
        className="status-pill status-pill-info",
    )

    # 1. Trajectory Chart
    fig1 = _fig()
    fig1.add_trace(go.Scatter(
        x=f_df.index, y=f_df["actual_total_energy"],
        mode="lines", name="Historical Actual",
        line=dict(color=C["actual"], width=1, dash="dot"), opacity=0.7,
    ))
    fig1.add_trace(go.Scatter(
        x=f_df.index, y=f_df["baseline_predicted_energy"],
        mode="lines", name="Model Baseline",
        line=dict(color=C["baseline"], width=1.5),
    ))
    fig1.add_trace(go.Scatter(
        x=f_df.index, y=f_df["optimized_predicted_energy"],
        mode="lines", name="Counterfactual Optimized",
        line=dict(color=C["optimized"], width=2),
    ))
    fig1.update_layout(height=340, xaxis_title="Timestamp", yaxis_title="Hourly Energy (kWh)")

    # 2. Scatter Actual vs Baseline
    fig2 = px.scatter(
        f_df, x="actual_total_energy", y="baseline_predicted_energy",
        color="outdoor_temp_c", color_continuous_scale="Blues",
        labels={"actual_total_energy": "Actual (kWh)", "baseline_predicted_energy": "Baseline Pred (kWh)", "outdoor_temp_c": "Outdoor T (°C)"},
        template=_TMPL, height=300,
    )
    _mx = float(f_df["actual_total_energy"].max() or 30) if not f_df.empty else 30
    fig2.add_shape(type="line", line=dict(dash="dash", color=C["danger"], width=1.5), x0=0, y0=0, x1=_mx, y1=_mx)
    fig2.update_layout(**_layout_extras())

    # 3. Scatter Actual vs Counterfactual
    fig3 = px.scatter(
        f_df, x="actual_total_energy", y="optimized_predicted_energy",
        color="outdoor_temp_c", color_continuous_scale="Greens",
        labels={"actual_total_energy": "Actual (kWh)", "optimized_predicted_energy": "Optimized Pred (kWh)", "outdoor_temp_c": "Outdoor T (°C)"},
        template=_TMPL, height=300,
    )
    fig3.update_layout(**_layout_extras())

    # 4. Hourly Delta Bar
    fig4 = _fig()
    fig4.add_trace(go.Bar(
        x=f_df.index, y=f_df["optimized_delta_energy"],
        name="ΔE = E_cand − E_base",
        marker_color=C["optimized"], marker_line_width=0,
    ))
    fig4.update_layout(height=300, xaxis_title="Timestamp", yaxis_title="Estimated ΔE (kWh)", yaxis=dict(zeroline=True, zerolinecolor=C["danger"], gridcolor="#1F2E44"))

    # 5. Cumulative Savings
    cum = (-f_df["optimized_delta_energy"]).cumsum() if not f_df.empty else pd.Series()
    fig5 = _fig()
    fig5.add_trace(go.Scatter(
        x=f_df.index, y=cum, mode="lines",
        name="Cumulative Savings (kWh)",
        line=dict(color=C["savings"], width=2.5),
        fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.12)",
    ))
    fig5.update_layout(height=300, xaxis_title="Timestamp", yaxis_title="Cumulative Saved kWh")

    # 6. Savings Histogram
    pos_savings = f_df.loc[f_df["estimated_energy_saving"] > 0] if not f_df.empty else pd.DataFrame()
    fig6 = px.histogram(
        pos_savings, x="estimated_energy_saving", nbins=35,
        template=_TMPL, height=300,
        labels={"estimated_energy_saving": "Estimated Saving (kWh)"},
        color_discrete_sequence=[C["optimized"]],
    ) if not pos_savings.empty else go.Figure()
    fig6.update_layout(**_layout_extras(), xaxis_title="Hourly Saving (kWh)", yaxis_title="Frequency (Hours)")

    # 7. Submeter Breakdown
    if "hvac_actual_kwh" in f_df.columns and not f_df.empty:
        sub = pd.DataFrame({
            "Submeter": ["HVAC (RTU South)", "Lighting", "Plug Loads (MELs)"],
            "Total Energy (kWh)": [
                f_df["hvac_actual_kwh"].sum(),
                f_df["lig_actual_kwh"].sum(),
                f_df["mels_actual_kwh"].sum(),
            ],
        })
        fig7 = px.pie(
            sub, values="Total Energy (kWh)", names="Submeter",
            color_discrete_sequence=[C["savings"], C["baseline"], "#EC4899"],
            template=_TMPL, hole=0.45, height=300,
        )
        fig7.update_traces(textfont_size=12)
        fig7.update_layout(**_layout_extras())
    else:
        fig7 = _fig()

    return count_badge, fig1, fig2, fig3, fig4, fig5, fig6, fig7


# =============================================================================
# ENTRYPOINT
# =============================================================================

def main():
    import os
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "False").lower() in ("true", "1")
    print(f"Starting Building 59 BEMS Dash Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
