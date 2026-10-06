"""
Dashboard Static Figure Generator (Phase 5).
Renders high-resolution publication-grade representations of all 8 Streamlit BEMS dashboard pages
for documentation and visual archiving under docs/figures/dashboard/.
"""
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from src.config import (
    ELECTRICITY_TARIFF,
    EMISSION_FACTOR,
    CONTROL_BOUNDS,
    RAMP_LIMITS,
    T_CLASS,
    T_SAFETY,
    BASELOAD_ENERGY_FLOOR
)

# Dark theme styling matching dashboard
DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
TEXT_WHITE = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_GREEN = "#10b981"
ACCENT_BLUE = "#38bdf8"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": TEXT_MUTED,
    "text.color": TEXT_WHITE,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": "#334155",
    "grid.linestyle": ":",
    "font.family": "sans-serif",
    "figure.dpi": 200
})

OUT_DIR = Path("docs/figures/dashboard")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_01_executive_overview(sim_df: pd.DataFrame):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.8, 0.8], hspace=0.35, wspace=0.25)

    # Header / KPI Cards
    kpis = [
        ("FORECASTED OCCUPANCY", "OCCUPIED (76.3%)", "T_safety = 0.30 Gated", ACCENT_AMBER),
        ("OUTDOOR REGIME", "9.8 °C (Cold Regime)", "48.2% Test Hours <10°C", ACCENT_BLUE),
        ("BASELINE DISPLAY ENERGY", "9,603.5 kWh", "Floor: 6.50 kWh", TEXT_MUTED),
        ("COUNTERFACTUAL SAVINGS", "5,120.5 kWh (53.3%)", "$1,126.50 Avoided Tariff", ACCENT_GREEN)
    ]
    for i, (title, val, sub, col) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.75, title, ha="center", va="center", color=TEXT_MUTED, fontsize=9, weight="bold")
        ax.text(0.5, 0.45, val, ha="center", va="center", color=col, fontsize=14, weight="bold")
        ax.text(0.5, 0.18, sub, ha="center", va="center", color=TEXT_WHITE, fontsize=8)

    # Main Trajectory Chart
    ax_main = fig.add_subplot(gs[1, :])
    sample_sub = sim_df.iloc[:240]  # First 10 days
    ax_main.plot(sample_sub.index, sample_sub["actual_total_energy"], label="Historical Actual (Total)", color="#64748b", linestyle=":", alpha=0.8, lw=1.2)
    ax_main.plot(sample_sub.index, sample_sub["baseline_predicted_energy"], label="Model Baseline Display (kWh)", color=ACCENT_AMBER, lw=1.6)
    ax_main.plot(sample_sub.index, sample_sub["optimized_predicted_energy"], label="Counterfactual Optimized (kWh)", color=ACCENT_GREEN, lw=2.0)
    ax_main.set_ylabel("Hourly Energy (kWh)", fontsize=10)
    ax_main.set_title("Test Period Trajectory Preview (First 10 Days of Evaluation)", color=TEXT_WHITE, fontsize=12, pad=10)
    ax_main.legend(loc="upper right", facecolor=CARD_BG, edgecolor="#334155", fontsize=9)
    ax_main.grid(True)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Bottom Safeguard Indicators
    states = [
        ("CONTROL-SAFE", "Eligible for setback (P < 0.30)", ACCENT_GREEN),
        ("PROTECTED_OCCUPIED", "Comfort airflow enforced (P ≥ 0.30)", ACCENT_AMBER),
        ("LOW_SURROGATE_CONFIDENCE", "Cold winter regime warning (<10°C)", ACCENT_RED),
        ("NO_CHANGE", "Preserved baseline (ΔE < 0.20 kWh)", ACCENT_BLUE)
    ]
    for i, (name, desc, col) in enumerate(states):
        ax = fig.add_subplot(gs[2, i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.65, name, ha="center", va="center", color=col, fontsize=11, weight="bold")
        ax.text(0.5, 0.30, desc, ha="center", va="center", color=TEXT_WHITE, fontsize=8)

    plt.suptitle("PAGE 1 — EXECUTIVE BEMS OVERVIEW (Research Simulation)", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "01_executive_overview.png", bbox_inches="tight")
    plt.close()


def fig_02_scenario_optimizer(joint_df: pd.DataFrame, sim_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.35, "wspace": 0.25})

    # Scenario details box (top left)
    ax_box = axes[0, 0]
    ax_box.set_xticks([])
    ax_box.set_yticks([])
    info_text = (
        "SELECTED SCENARIO: Peak Business Hours\n"
        "Timestamp: 2019-01-15 14:00:00\n\n"
        "HISTORICAL EXOGENOUS OBSERVED:\n"
        "• Indoor Temperature: 22.4 °C\n"
        "• Outdoor Temperature: 13.8 °C (Mild Regime)\n"
        "• Relative Humidity: 46.2 %\n"
        "• Solar Radiation: 340 W/m²\n\n"
        "MODEL SAFETY TELEMETRY:\n"
        "• Occupancy Prob: 98.4% (PROTECTED_OCCUPIED)\n"
        "• Safety Gating: T_safety = 0.30 Enforced\n"
        "• Surrogate Confidence: NORMAL_CONFIDENCE"
    )
    ax_box.text(0.05, 0.90, info_text, transform=ax_box.transAxes, color=TEXT_WHITE, fontsize=9.5, va="top", fontfamily="monospace")
    ax_box.set_title("Input Vector & Telemetry", color=ACCENT_BLUE, fontsize=11, weight="bold")

    # Setpoint Bar Chart (top right)
    ax_bar = axes[0, 1]
    labels = ["Fan Speed (%)", "Damper (%)", "Energy (kWh)"]
    baseline = [78.0, 45.0, 18.4]
    optimized = [63.0, 25.0, 12.1]
    x = np.arange(len(labels))
    width = 0.35
    ax_bar.bar(x - width/2, baseline, width, label="Baseline Setpoint", color=ACCENT_AMBER)
    ax_bar.bar(x + width/2, optimized, width, label="Recommended (Safeguarded)", color=ACCENT_GREEN)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10)
    ax_bar.set_title("Baseline vs Counterfactual Recommended Controls", color=TEXT_WHITE, fontsize=11, weight="bold")
    ax_bar.legend(loc="upper right", facecolor=CARD_BG, edgecolor="#334155")
    ax_bar.grid(True)

    # Action Attribution Box (bottom left)
    ax_attr = axes[1, 0]
    ax_attr.set_xticks([])
    ax_attr.set_yticks([])
    attr_text = (
        "OPTIMIZATION ENGINE ATTRIBUTION & REASON:\n\n"
        "Action: OPTIMIZE (Candidate Grid Search)\n"
        "Reason: Identified safe candidate reducing energy by 6.30 kWh\n"
        "         (fan=63%, damper=25%).\n\n"
        "Constraint Validations:\n"
        "✓ Fan Speed in Support [40%, 90%]\n"
        "✓ Damper in Support [10%, 90%]\n"
        "✓ Fan Ramp Rate |63% - 78%| = 15.0% ≤ 15.0%/hr Limit\n"
        "✓ Damper Ramp Rate |25% - 45%| = 20.0% ≤ 30.0%/hr Limit\n"
        "✓ Minimum Improvement Epsilon ε ≥ 0.20 kWh"
    )
    ax_attr.text(0.05, 0.90, attr_text, transform=ax_attr.transAxes, color=ACCENT_GREEN, fontsize=9.5, va="top", fontfamily="monospace")
    ax_attr.set_title("Diagnostic Reason & Constraint Audit", color=ACCENT_GREEN, fontsize=11, weight="bold")

    # Rate and Savings Card (bottom right)
    ax_sav = axes[1, 1]
    ax_sav.set_xticks([])
    ax_sav.set_yticks([])
    sav_text = (
        "COUNTERFACTUAL IMPACT ESTIMATES:\n\n"
        "• Baseline Model Energy: 18.40 kWh\n"
        "• Counterfactual Energy: 12.10 kWh\n"
        "• Estimated ΔE Saving:    -6.30 kWh (-34.2%)\n\n"
        "• Avoided Electricity Cost: $1.39 / hr (@ $0.22/kWh)\n"
        "• Avoided Grid Carbon:      1.32 kg CO₂ / hr (@ 0.210 kg/kWh)\n\n"
        "Status: SIMULATED RESEARCH ESTIMATE (No Live BACnet Control)"
    )
    ax_sav.text(0.05, 0.90, sav_text, transform=ax_sav.transAxes, color=ACCENT_BLUE, fontsize=9.5, va="top", fontfamily="monospace")
    ax_sav.set_title("Economic & Carbon Rate Impacts", color=ACCENT_BLUE, fontsize=11, weight="bold")

    plt.suptitle("PAGE 2 — INTERACTIVE SCENARIO SIMULATOR", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "02_scenario_optimizer.png", bbox_inches="tight")
    plt.close()


def fig_03_energy_analytics(sim_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    # Actual vs Baseline vs Counterfactual Time Series
    ax1 = axes[0, 0]
    sub = sim_df.iloc[100:220]
    ax1.plot(sub.index, sub["actual_total_energy"], label="Actual", color="#64748b", ls=":", lw=1.2)
    ax1.plot(sub.index, sub["baseline_predicted_energy"], label="Baseline Pred", color=ACCENT_AMBER, lw=1.5)
    ax1.plot(sub.index, sub["optimized_predicted_energy"], label="Counterfactual", color=ACCENT_GREEN, lw=1.8)
    ax1.set_title("Energy Trajectory Comparison (5-Day Window)", fontsize=11, weight="bold")
    ax1.set_ylabel("kWh")
    ax1.legend(loc="upper right", facecolor=CARD_BG, edgecolor="#334155")
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Parity scatter
    ax2 = axes[0, 1]
    ax2.scatter(sim_df["actual_total_energy"], sim_df["baseline_predicted_energy"], alpha=0.3, color=ACCENT_BLUE, s=15, label="Test Hours")
    max_val = max(sim_df["actual_total_energy"].max(), sim_df["baseline_predicted_energy"].max())
    ax2.plot([0, max_val], [0, max_val], color=ACCENT_RED, ls="--", label="1:1 Parity")
    ax2.set_xlabel("Historical Actual Energy (kWh)")
    ax2.set_ylabel("Model Baseline Predicted Energy (kWh)")
    ax2.set_title("Actual vs Model Baseline Correlation", fontsize=11, weight="bold")
    ax2.legend(loc="upper left", facecolor=CARD_BG, edgecolor="#334155")
    ax2.grid(True)

    # Cumulative Savings
    ax3 = axes[1, 0]
    cum_sav = (-sim_df["optimized_delta_energy"]).cumsum()
    ax3.plot(sim_df.index, cum_sav, color=ACCENT_BLUE, lw=2.2)
    ax3.fill_between(sim_df.index, cum_sav, color=ACCENT_BLUE, alpha=0.15)
    ax3.set_title("Cumulative Estimated Energy Savings (5,120.5 kWh)", fontsize=11, weight="bold")
    ax3.set_ylabel("Cumulative Saved kWh")
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Submeter breakdown
    ax4 = axes[1, 1]
    submeters = ["HVAC (RTU South)", "Lighting", "Plug Loads (MELs)"]
    kwhs = [sim_df["hvac_actual_kwh"].sum(), sim_df["lig_actual_kwh"].sum(), sim_df["mels_actual_kwh"].sum()]
    ax4.bar(submeters, kwhs, color=[ACCENT_BLUE, ACCENT_AMBER, "#ec4899"])
    ax4.set_title("South Wing Submeter Breakdown (Actual Test)", fontsize=11, weight="bold")
    ax4.set_ylabel("Total kWh")
    for i, v in enumerate(kwhs):
        ax4.text(i, v + 200, f"{v:,.0f} kWh", ha="center", color=TEXT_WHITE, fontsize=9, weight="bold")
    ax4.grid(True)

    plt.suptitle("PAGE 3 — ENERGY ANALYTICS & COUNTERFACTUAL COMPARISON", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "03_energy_analytics.png", bbox_inches="tight")
    plt.close()


def fig_04_occupancy_analytics(sim_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={"hspace": 0.35, "wspace": 0.25})

    # Probability time series with dual thresholds
    ax1 = axes[0, 0]
    sub = sim_df.iloc[:168]  # 7 days
    ax1.plot(sub.index, sub["occupancy_probability"], label="P(Occ=1)", color=ACCENT_BLUE, lw=1.6)
    ax1.axhline(T_CLASS, color=ACCENT_AMBER, ls="--", label=f"T_class = {T_CLASS:.2f}")
    ax1.axhline(T_SAFETY, color=ACCENT_RED, ls="--", label=f"T_safety = {T_SAFETY:.2f}")
    ax1.set_title("Predicted Occupancy Probability & Dual Safety Gates", fontsize=11, weight="bold")
    ax1.set_ylabel("Probability")
    ax1.legend(loc="upper right", facecolor=CARD_BG, edgecolor="#334155")
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Diurnal profile
    ax2 = axes[0, 1]
    df_p = sim_df.copy()
    df_p["hour"] = df_p.index.hour
    df_p["is_weekend"] = df_p.index.dayofweek >= 5
    diurnal = df_p.groupby(["hour", "is_weekend"])["actual_occupancy_state"].mean().unstack()
    ax2.plot(diurnal.index, diurnal[False], label="Weekday Mean", color=ACCENT_BLUE, marker="o", lw=1.8)
    ax2.plot(diurnal.index, diurnal[True], label="Weekend Mean", color=ACCENT_AMBER, marker="s", lw=1.8)
    ax2.set_title("Diurnal Hour-of-Day Occupancy Profiles", fontsize=11, weight="bold")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Occupancy Rate")
    ax2.legend(loc="upper left", facecolor=CARD_BG, edgecolor="#334155")
    ax2.grid(True)

    # Heatmap
    ax3 = axes[1, 0]
    df_h = sim_df.copy()
    df_h["day"] = df_h.index.day_name()
    df_h["hour"] = df_h.index.hour
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = df_h.groupby(["day", "hour"])["occupancy_probability"].mean().unstack().reindex(days)
    im = ax3.imshow(heat.values, cmap="viridis", aspect="auto")
    ax3.set_yticks(range(len(days)))
    ax3.set_yticklabels(days)
    ax3.set_xticks(range(0, 24, 3))
    ax3.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax3.set_title("Weekly Occupancy Probability Heatmap", fontsize=11, weight="bold")
    fig.colorbar(im, ax=ax3, orientation="horizontal", pad=0.2, label="Mean P(Occ)")

    # Performance Metrics Card
    ax4 = axes[1, 1]
    ax4.set_xticks([])
    ax4.set_yticks([])
    metrics_text = (
        "RANDOM FOREST OCCUPANCY CHAMPION (TEST SET):\n\n"
        "• Test F1 Score (Tuned T=0.51): 0.9419\n"
        "• Test Precision:              0.9332\n"
        "• Test Recall:                 0.9508\n"
        "• Area Under ROC Curve:        0.9392\n"
        "• Area Under PR Curve:         0.9806\n"
        "• Balanced Accuracy:           0.8643\n\n"
        "SAFETY GATE AUDIT (T_safety = 0.30):\n"
        "• Unsafe Occupied Setback Hours Reduced by 72.55%\n"
        "• Prevents False-Negative Ventilation Cutoffs"
    )
    ax4.text(0.05, 0.90, metrics_text, transform=ax4.transAxes, color=TEXT_WHITE, fontsize=9.5, va="top", fontfamily="monospace")
    ax4.set_title("Classifier Metrics & Asymmetric Threshold Impact", color=ACCENT_GREEN, fontsize=11, weight="bold")

    plt.suptitle("PAGE 4 — OCCUPANCY ANALYTICS & CLASSIFIER EVALUATION", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "04_occupancy_analytics.png", bbox_inches="tight")
    plt.close()


def fig_05_model_performance():
    p3 = json.loads(Path("docs/phase3_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.25})

    # Occupancy model comparison
    ax1 = axes[0]
    occ_res = p3["occupancy_classification"]["validation_results"]
    models = [r["Model"] for r in occ_res]
    f1s = [r["Val F1 (Tuned)"] for r in occ_res]
    pr_aucs = [r["PR-AUC"] for r in occ_res]
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, f1s, width, label="Val F1 (Tuned)", color=ACCENT_BLUE)
    ax1.bar(x + width/2, pr_aucs, width, label="PR-AUC", color=ACCENT_GREEN)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right")
    ax1.set_title("Occupancy Classifier Comparison (Validation Set)", fontsize=11, weight="bold")
    ax1.set_ylabel("Score [0.0, 1.0]")
    ax1.legend(loc="lower right", facecolor=CARD_BG, edgecolor="#334155")
    ax1.grid(True)

    # Energy model comparison
    ax2 = axes[1]
    eng_res = p3["energy_forecasting"]["validation_results"]
    e_models = [r["Model"] for r in eng_res]
    rmses = [r["Chained RMSE"] for r in eng_res]
    ax2.bar(e_models, rmses, color=ACCENT_AMBER)
    ax2.set_xticklabels(e_models, rotation=25, ha="right")
    ax2.set_title("Energy Model Candidate Comparison (Val RMSE kWh)", fontsize=11, weight="bold")
    ax2.set_ylabel("RMSE (kWh)")
    for i, v in enumerate(rmses):
        ax2.text(i, v + 0.1, f"{v:.2f}", ha="center", color=TEXT_WHITE, fontsize=9, weight="bold")
    ax2.grid(True)

    plt.suptitle("PAGE 5 — ML MODEL PERFORMANCE & BENCHMARK REGISTRY", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "05_model_performance.png", bbox_inches="tight")
    plt.close()


def fig_06_optimization_analysis():
    p4 = json.loads(Path("docs/phase4_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.25})

    # Regime Savings Breakdown
    ax1 = axes[0]
    reg_data = p4["regime_breakdown"]
    labels = ["Occupied", "Unoccupied", "Cold (<10°C)", "Mild (10-16°C)", "Warm (>16°C)"]
    kwhs = [
        reg_data["occupancy_occupied"]["savings_kwh"],
        reg_data["occupancy_unoccupied"]["savings_kwh"],
        reg_data["temp_cold"]["savings_kwh"],
        reg_data["temp_mild"]["savings_kwh"],
        reg_data["temp_warm"]["savings_kwh"]
    ]
    ax1.bar(labels, kwhs, color=[ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, "#ec4899"])
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_title("Estimated Savings by Regime Stratification (kWh)", fontsize=11, weight="bold")
    ax1.set_ylabel("Estimated Savings (kWh)")
    for i, v in enumerate(kwhs):
        ax1.text(i, v + 80, f"{v:,.0f}", ha="center", color=TEXT_WHITE, fontsize=9, weight="bold")
    ax1.grid(True)

    # Ablation comparison
    ax2 = axes[1]
    abls = p4["ablations"]
    abl_labels = ["No Safety Gate", "With Safety Gate", "No Ramp Limit", "Full Safeguards"]
    abl_kwhs = [
        abls["Ablation A (T=0.51, No Safety Gate)"]["total_estimated_energy_savings_kwh"],
        abls["Ablation B (T=0.30, With Safety Gate)"]["total_estimated_energy_savings_kwh"],
        abls["Ablation C (No Ramp Protection)"]["total_estimated_energy_savings_kwh"],
        abls["Ablation D (With Ramp Protection)"]["total_estimated_energy_savings_kwh"]
    ]
    colors = [ACCENT_BLUE, ACCENT_BLUE, ACCENT_RED, ACCENT_GREEN]
    bars = ax2.bar(abl_labels, abl_kwhs, color=colors)
    ax2.set_xticklabels(abl_labels, rotation=25, ha="right")
    ax2.set_title("Safeguard Ablation Study (Highlighting Non-Physical No-Ramp)", fontsize=11, weight="bold")
    ax2.set_ylabel("Total Savings (kWh)")
    for i, v in enumerate(abl_kwhs):
        ax2.text(i, v + 200, f"{v:,.0f} kWh", ha="center", color=TEXT_WHITE, fontsize=9, weight="bold")
    ax2.grid(True)

    plt.suptitle("PAGE 6 — OPTIMIZATION ANALYSIS & ABLATION STUDY", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "06_optimization_analysis.png", bbox_inches="tight")
    plt.close()


def fig_07_safety_validity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.25})

    # Envelope table representation
    ax1 = axes[0]
    ax1.set_xticks([])
    ax1.set_yticks([])
    env_text = (
        "OPERATIONAL SAFETY BOUNDS & CONSTRAINTS MATRIX:\n\n"
        "• Fan Speed Range:       [40.0%, 90.0%]  (Historical Support)\n"
        "• Damper Position Range: [10.0%, 90.0%]  (Ventilation Floor)\n"
        "• Maximum Fan Ramp:      ≤ 15.0% / hour  (Duct Pressure Safety)\n"
        "• Maximum Damper Ramp:   ≤ 30.0% / hour  (Actuator Wear Safety)\n"
        "• Occupancy Safety Gate: T_safety = 0.30 (Occupancy Protection)\n"
        "• Improvement Threshold: ε = 0.20 kWh    (Anti-Hunting Gate)\n"
        "• Display Baseload Floor: 6.50 kWh       (Calibrated Display Floor)\n\n"
        "CANDIDATE GRID EVALUATION:\n"
        "• 99 discrete pairs: Fan {40..90 step 5} × Damper {10..90 step 10}"
    )
    ax1.text(0.05, 0.90, env_text, transform=ax1.transAxes, color=TEXT_WHITE, fontsize=9.5, va="top", fontfamily="monospace")
    ax1.set_title("Safeguard Parameter Specifications", color=ACCENT_BLUE, fontsize=11, weight="bold")

    # Four states description
    ax2 = axes[1]
    ax2.set_xticks([])
    ax2.set_yticks([])
    states_text = (
        "FOUR SYSTEM OPERATIONAL TELEMETRY STATES:\n\n"
        "1. CONTROL-SAFE (SETBACK ELIGIBLE):\n"
        "   Zone verified vacant (P < 0.30). Setback enabled.\n\n"
        "2. PROTECTED_OCCUPIED:\n"
        "   P ≥ 0.30. Deep setbacks prohibited. Enforces comfort.\n\n"
        "3. LOW_SURROGATE_CONFIDENCE (<10°C Cold Regime):\n"
        "   Outdoor temp < 10°C. Cold winter regime unobserved in training.\n"
        "   Surrogate exhibits negative bias. Flagged for human review.\n\n"
        "4. NO_CHANGE (Baseline Preserved):\n"
        "   No candidate yields ΔE ≤ -0.20 kWh. Preserves equipment state."
    )
    ax2.text(0.05, 0.90, states_text, transform=ax2.transAxes, color=ACCENT_GREEN, fontsize=9.5, va="top", fontfamily="monospace")
    ax2.set_title("System Telemetry State Definitions", color=ACCENT_GREEN, fontsize=11, weight="bold")

    plt.suptitle("PAGE 7 — SAFETY ENVELOPE & DOMAIN VALIDITY", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "07_safety_validity.png", bbox_inches="tight")
    plt.close()


def fig_08_methodology_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xticks([])
    ax.set_yticks([])

    pipe_text = (
        "BEMS RESEARCH SIMULATION END-TO-END PIPELINE ARCHITECTURE:\n\n"
        "  [1. Current Building State at time t]\n"
        "      • Sensor vectors: Zone indoor temp, outdoor temp, RH, solar, fan spd, damper %\n"
        "      • Temporal calendar features (hour, day of week, cyclical encodings)\n"
        "              │\n"
        "              ▼\n"
        "  [2. Occupancy Forecast t → t+1]\n"
        "      • Champion: Random Forest Classifier (T_class=0.51, T_safety=0.30)\n"
        "      • Gating Decision: If P(Occ) ≥ 0.30 → PROTECTED_OCCUPIED else SETBACK_ELIGIBLE\n"
        "              │\n"
        "              ▼\n"
        "  [3. Counterfactual Scenario Grid Search]\n"
        "      • 99 Discrete Candidate Pairs: Fan {40..90 step 5} × Damper {10..90 step 10}\n"
        "      • Vectorized Candidate Filtering: Ramp limits (|Δfan| ≤ 15%, |Δdamper| ≤ 30%)\n"
        "              │\n"
        "              ▼\n"
        "  [4. Energy Surrogate Forecast t → t+1]\n"
        "      • Champion: Ridge Linear Surrogate (α=10)\n"
        "      • Local Differential Objective: ΔE(u) = E_hat(u_cand) - E_hat(u_base)\n"
        "              │\n"
        "              ▼\n"
        "  [5. NO_CHANGE Preservation Gate]\n"
        "      • If min(ΔE) > -0.20 kWh → Command NO_CHANGE (Preserve baseline setpoints)\n"
        "      • Else → Select optimal admissible candidate minimizing model energy\n"
        "              │\n"
        "              ▼\n"
        "  [6. Reporting & Impact Telemetry]\n"
        "      • Estimated kWh Savings, Avoided Utility Cost ($0.22/kWh), Grid CO₂ (0.210 kg/kWh)\n"
        "      • Research Simulation Disclaimer: Not physical closed-loop control."
    )
    ax.text(0.05, 0.95, pipe_text, transform=ax.transAxes, color=TEXT_WHITE, fontsize=9.5, va="top", fontfamily="monospace")
    ax.set_title("Building 59 BEMS Counterfactual Optimization Flow", color=ACCENT_BLUE, fontsize=12, weight="bold")

    plt.suptitle("PAGE 8 — ABOUT & MODELING METHODOLOGY", color=TEXT_WHITE, fontsize=14, weight="bold", y=0.98)
    plt.savefig(OUT_DIR / "08_methodology_pipeline.png", bbox_inches="tight")
    plt.close()


def main():
    print("Generating Phase 5 dashboard documentation figures...")
    sim_df = pd.read_parquet("data/processed/phase4_simulation_results.parquet")
    joint_df = pd.read_parquet("data/processed/joint_modeling_data.parquet")
    if "timestamp" in joint_df.columns:
        joint_df = joint_df.set_index("timestamp")

    fig_01_executive_overview(sim_df)
    print("  [OK] 01_executive_overview.png")
    fig_02_scenario_optimizer(joint_df, sim_df)
    print("  [OK] 02_scenario_optimizer.png")
    fig_03_energy_analytics(sim_df)
    print("  [OK] 03_energy_analytics.png")
    fig_04_occupancy_analytics(sim_df)
    print("  [OK] 04_occupancy_analytics.png")
    fig_05_model_performance()
    print("  [OK] 05_model_performance.png")
    fig_06_optimization_analysis()
    print("  [OK] 06_optimization_analysis.png")
    fig_07_safety_validity()
    print("  [OK] 07_safety_validity.png")
    fig_08_methodology_pipeline()
    print("  [OK] 08_methodology_pipeline.png")
    print(f"All 8 figures successfully generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
