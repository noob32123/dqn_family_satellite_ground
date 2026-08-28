"""Draw the family-level DQN versus MPC-4 evidence summary in Python."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "reviewer_revision_state_complete"
OUTPUT = ROOT.parent / "reproduced_outputs" / "figures" / "fig_dqn_family_summary"

SCENARIOS = (
    "nominal",
    "burst",
    "link_limited",
    "energy_limited",
    "thermal_stress",
)
POLICIES = (
    "standard_dqn",
    "full_action_q_dqn",
    "immediate_advantage_dqn",
    "centered_full_action_dqn",
    "double_dqn",
    "double_centered_full_action_dqn",
)
SCENARIO_LABELS = {
    "nominal": "Nominal",
    "burst": "Burst",
    "link_limited": "Link-limited",
    "energy_limited": "Energy-limited",
    "thermal_stress": "Thermal-stress",
}
POLICY_LABELS = {
    "standard_dqn": "Standard\nDQN",
    "full_action_q_dqn": "Full-action\nQ auxiliary",
    "immediate_advantage_dqn": "Immediate\nadvantage",
    "centered_full_action_dqn": "Centered\nfull-action",
    "double_dqn": "Double\nDQN",
    "double_centered_full_action_dqn": "Double +\ncentered",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "axes.titlesize": 7.3,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _flow_panel(axis: plt.Axes) -> None:
    axis.set_axis_off()
    _panel_label(axis, "a")
    boxes = (
        (0.01, 0.16, 0.22, 0.68, "Synthetic soft-constrained\nresource-coupled benchmark", "#EEF2F4", "#40505A"),
        (0.28, 0.16, 0.18, 0.68, "Six DQN objectives\nand exact MPC-4", "#DCEFF1", "#007C91"),
        (0.51, 0.16, 0.20, 0.68, "20 model-seed blocks\n× 20 shared test traces", "#E7EEF6", "#315B7D"),
        (0.76, 0.16, 0.23, 0.68, "Paired Δ = DQN − MPC-4\nJoint seed-block bootstrap", "#FCEBD8", "#C66A1B"),
    )
    for x, y, width, height, label, face, edge in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=6.4,
            color="#1E2A31",
            linespacing=1.25,
        )
    for left, right in ((0.23, 0.28), (0.46, 0.51), (0.71, 0.76)):
        axis.add_patch(
            FancyArrowPatch(
                (left + 0.006, 0.50),
                (right - 0.006, 0.50),
                transform=axis.transAxes,
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.9,
                color="#6B7780",
            )
        )


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    cells = pd.read_csv(RESULTS / "dqn_vs_mpc_inference.csv")
    maxima = pd.read_csv(RESULTS / "dqn_vs_mpc_family_inference.csv")
    expected = pd.MultiIndex.from_product(
        [SCENARIOS, POLICIES], names=["scenario", "policy"]
    )
    ordered = (
        cells.set_index(["scenario", "policy"])
        .reindex(expected)
        .reset_index()
    )
    if len(ordered) != 30 or ordered.mean_difference.isna().any():
        raise AssertionError("the family figure requires 30 complete contrasts")
    if not ordered.mean_difference.lt(0).all():
        raise AssertionError("the plotted conclusion requires every mean difference below zero")
    matrix = ordered.mean_difference.to_numpy().reshape(len(SCENARIOS), len(POLICIES))
    return ordered, maxima, matrix


def _heatmap_panel(axis: plt.Axes, matrix: np.ndarray) -> None:
    _panel_label(axis, "b")
    cmap = LinearSegmentedColormap.from_list(
        "dqn_advantage", ("#007C91", "#8CC7CD", "#F1F7F7")
    )
    image = axis.imshow(
        matrix,
        cmap=cmap,
        vmin=-14.5,
        vmax=-5.5,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(range(len(POLICIES)), [POLICY_LABELS[item] for item in POLICIES])
    axis.set_yticks(range(len(SCENARIOS)), [SCENARIO_LABELS[item] for item in SCENARIOS])
    axis.tick_params(axis="x", length=0, pad=3)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.set_title(
        "Mean episode-cost difference for every DQN objective and regime",
        loc="left",
        pad=6,
        fontweight="bold",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value <= -10.8 else "#17242B"
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=5.8,
                fontweight="bold",
            )
    axis.set_xticks(np.arange(-0.5, len(POLICIES), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(SCENARIOS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = axis.figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        fraction=0.08,
        pad=0.18,
        aspect=30,
    )
    colorbar.set_label("Δ mean episode cost (DQN − MPC-4); lower favors DQN")
    colorbar.outline.set_linewidth(0.5)


def _maximum_panel(axis: plt.Axes, maxima: pd.DataFrame) -> None:
    _panel_label(axis, "c")
    order = list(SCENARIOS) + ["all_fully_coupled"]
    selected = maxima.set_index("scope").loc[order].reset_index()
    y = np.arange(len(selected))
    means = selected.maximum_mean_difference.to_numpy()
    lower = selected.max_statistic_ci95_low.to_numpy()
    upper = selected.max_statistic_ci95_high.to_numpy()
    bounds = selected.simultaneous_95_upper_bound.to_numpy()
    colors = ["#007C91"] * len(SCENARIOS) + ["#C66A1B"]
    for index in range(len(selected)):
        axis.errorbar(
            means[index],
            y[index],
            xerr=np.array([[means[index] - lower[index]], [upper[index] - means[index]]]),
            fmt="o",
            markersize=4.2,
            markerfacecolor=colors[index],
            markeredgecolor="white",
            markeredgewidth=0.5,
            ecolor=colors[index],
            elinewidth=1.1,
            capsize=2.2,
            capthick=0.8,
            zorder=3,
        )
        axis.plot(
            bounds[index],
            y[index],
            marker=">",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=colors[index],
            markeredgewidth=0.9,
            linestyle="none",
            zorder=4,
        )
    labels = [SCENARIO_LABELS[item] for item in SCENARIOS] + ["All 30 contrasts"]
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    axis.set_xlim(-15.0, 0.8)
    axis.set_xlabel("Least-favorable mean Δ within scope")
    axis.set_title(
        "Worst member remains below MPC-4",
        loc="left",
        pad=6,
        fontweight="bold",
    )
    axis.grid(axis="x", color="#D9DEE2", linewidth=0.5, zorder=0)
    axis.tick_params(axis="y", length=0)
    axis.text(
        0.01,
        -0.25,
        "Circle and line: maximum observed mean Δ and its 95% bootstrap interval\n"
        "Open triangle: simultaneous 95% upper bound",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color="#40505A",
        linespacing=1.3,
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cells, maxima, matrix = _load_data()
    if not cells.holm_supported.all():
        raise AssertionError("all 30 contrasts must satisfy the prespecified support rule")
    figure = plt.figure(
        figsize=(7.2047, 4.4094),  # 183 mm × 112 mm
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.32, 1.0),
        width_ratios=(1.55, 1.0),
    )
    flow_axis = figure.add_subplot(grid[0, :])
    heatmap_axis = figure.add_subplot(grid[1, 0])
    maximum_axis = figure.add_subplot(grid[1, 1])
    _flow_panel(flow_axis)
    _heatmap_panel(heatmap_axis, matrix)
    _maximum_panel(maximum_axis, maxima)
    figure.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    figure.savefig(
        OUTPUT.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
