"""
Automated Unit and Integration Tests for Phase 5 Custom Dash BEMS Dashboard.
Verifies Dash application initialization, all 8 view layouts, reactive callbacks,
interactive filtering, scenario simulations, and CSS asset integration without browser overhead.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from dashboard.dash_app import (
    app,
    server,
    route_page,
    update_executive_overview,
    execute_scenario_simulator,
    update_energy_analytics,
    toggle_simulator_selection_mode,
    build_view_executive_overview,
    build_view_scenario_simulator,
    build_view_energy_analytics,
    build_view_occupancy_analytics,
    build_view_model_performance,
    build_view_optimization_analysis,
    build_view_safety_validity,
    build_view_about_methodology,
    NAV_ITEMS,
    rep_timestamps,
)


def test_dash_app_instance_and_metadata():
    """Verify Dash app initializes with expected configuration and title."""
    assert app is not None
    assert app.title == "Building 59 BEMS | AI Energy Optimization"
    assert server is not None
    assert hasattr(server, "route")  # Underlying Flask instance


def test_custom_css_assets_exist_and_valid():
    """Verify custom CSS stylesheet exists in dashboard/assets/ with required design tokens."""
    css_path = Path("dashboard/assets/custom.css")
    assert css_path.exists(), f"Missing stylesheet at {css_path}"
    
    css_content = css_path.read_text(encoding="utf-8")
    assert "--bg-primary" in css_content
    assert ".glass-card" in css_content
    assert ".kpi-card" in css_content
    assert ".status-pill" in css_content
    assert ".btn-bems-primary" in css_content


@pytest.mark.parametrize("builder_fn", [
    build_view_executive_overview,
    build_view_scenario_simulator,
    build_view_energy_analytics,
    build_view_occupancy_analytics,
    build_view_model_performance,
    build_view_optimization_analysis,
    build_view_safety_validity,
    build_view_about_methodology,
])
def test_all_eight_view_builders_execute_cleanly(builder_fn):
    """Verify that all 8 individual view builders return valid Dash component trees."""
    view = builder_fn()
    assert view is not None
    assert hasattr(view, "children")
    assert len(view.children) > 0


@pytest.mark.parametrize("item", NAV_ITEMS)
def test_router_callback_all_nav_routes(item):
    """Verify route_page correctly handles each route defined in navigation."""
    content, sidebar = route_page(item["href"])
    assert content is not None
    assert sidebar is not None
    assert hasattr(content, "children")
    assert hasattr(sidebar, "children")


def test_router_callback_404_fallback():
    """Verify unknown path returns graceful 404 message and fallback link."""
    content, sidebar = route_page("/unknown-nonexistent-path")
    assert content is not None
    assert "404" in str(content)


def test_executive_overview_callback_execution():
    """Verify executive overview callback produces all 12 expected component outputs."""
    sample_scenario = list(rep_timestamps.keys())[0]
    (
        badge, kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8,
        fig, rec_box, val_box
    ) = update_executive_overview(sample_scenario)

    assert badge is not None
    for kpi in (kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8):
        assert kpi is not None
        assert hasattr(kpi, "children")
    
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) >= 3  # Actual, baseline, optimized
    assert rec_box is not None
    assert val_box is not None


def test_scenario_simulator_toggle_mode():
    """Verify toggle between preset scenarios and custom timestamp dropdowns."""
    p_style, c_style = toggle_simulator_selection_mode("preset")
    assert p_style == {"display": "block"}
    assert c_style == {"display": "none"}

    p_style2, c_style2 = toggle_simulator_selection_mode("custom")
    assert p_style2 == {"display": "none"}
    assert c_style2 == {"display": "block"}


def test_scenario_simulator_callback_default_and_overrides():
    """Verify simulator callback runs optimization without and with overrides."""
    default_ts = rep_timestamps[list(rep_timestamps.keys())[0]]

    # 1. Baseline run (no overrides)
    sensor_cards, output_kpis, attribution_box, fig_bar = execute_scenario_simulator(
        n_clicks=1,
        preset_ts=default_ts,
        custom_ts="",
        mode="preset",
        use_overrides=False,
        ov_fan=65.0,
        ov_damper=30.0,
        ov_tin=22.0,
        ov_tout=12.0,
    )
    assert len(sensor_cards) == 6
    assert len(output_kpis) == 4
    assert attribution_box is not None
    assert fig_bar is not None
    assert len(fig_bar.data) == 2  # Baseline vs Recommended

    # 2. What-If Overrides run
    sensor_cards_ov, output_kpis_ov, attribution_box_ov, fig_bar_ov = execute_scenario_simulator(
        n_clicks=2,
        preset_ts=default_ts,
        custom_ts="",
        mode="preset",
        use_overrides=True,
        ov_fan=85.0,
        ov_damper=60.0,
        ov_tin=24.0,
        ov_tout=8.0,
    )
    assert len(sensor_cards_ov) == 6
    assert len(output_kpis_ov) == 4
    assert attribution_box_ov is not None


@pytest.mark.parametrize("occ_filter,temp_filter", [
    ("All", "All"),
    ("Occupied only", "Cold (<10 °C)"),
    ("Unoccupied only", "Mild (10-16 °C)"),
    ("All", "Warm (>16 °C)"),
])
def test_energy_analytics_filtering_callback(occ_filter, temp_filter):
    """Verify energy analytics callback correctly coordinates all 7 charts under filters."""
    badge, fig1, fig2, fig3, fig4, fig5, fig6, fig7 = update_energy_analytics(
        start_date="2019-01-11",
        end_date="2019-02-21",
        occ_filter=occ_filter,
        temp_filter=temp_filter,
    )
    assert badge is not None
    for fig in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        assert fig is not None
        assert hasattr(fig, "data")
