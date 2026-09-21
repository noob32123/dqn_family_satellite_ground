"""Generate second-round manuscript tables, figures, and an audit summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .revision_experiments import GAMMAS, TRAINING_LENGTHS, VARIANTS, curve_path, model_path


LABELS = {
    "standard_dqn": "Standard DQN",
    "full_action_q_dqn": "Full-action Q",
    "immediate_advantage_dqn": "Immediate advantage",
    "centered_full_action_dqn": "Centered full-action",
    "double_dqn": "Double DQN",
    "double_centered_full_action_dqn": "Double + centered",
    "mpc4": "MPC-4",
    "greedy8": "Greedy rollout-8",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLORS = dict(zip(VARIANTS, plt.get_cmap("tab10").colors[:len(VARIANTS)]))


def save_figure_bundle(figure: plt.Figure, output: Path) -> None:
    """Write editable vector and high-resolution raster companions."""
    figure.savefig(output, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def seed_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    by_seed = frame.groupby(groups + ["model_seed"], as_index=False).total_cost.mean()
    rows = []
    for index, selected in by_seed.groupby(groups, sort=True):
        index = index if isinstance(index, tuple) else (index,)
        values = selected.total_cost.to_numpy()
        low, high = bootstrap_ci(values, 20260829 + len(rows))
        rows.append({**dict(zip(groups, index)), "mean": values.mean(),
                     "sd": values.std(ddof=1), "ci_low": low, "ci_high": high,
                     "n": len(values)})
    return pd.DataFrame(rows)


def plot_learning_curves(results: Path, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), sharex=True, sharey=True)
    for axis, variant in zip(axes.flat, VARIANTS):
        curves = []
        for seed in range(800, 820):
            frame = pd.read_csv(curve_path(model_path(results, variant, .97, seed)))
            curves.append(frame.cost.rolling(25, min_periods=1).mean().to_numpy())
        matrix = np.stack(curves)
        episodes = np.arange(1, matrix.shape[1] + 1)
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        color = COLORS[variant]
        axis.plot(episodes, mean, color=color, lw=1.5)
        axis.fill_between(episodes, mean - sd, mean + sd, color=color, alpha=.18,
                          linewidth=0)
        axis.set_title(LABELS[variant], fontsize=9)
        axis.grid(axis="y", color="#D9D9D9", lw=.55)
        axis.tick_params(labelsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Training episode", fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("Smoothed episode cost", fontsize=9)
    figure.tight_layout(pad=.8)
    save_figure_bundle(figure, output)
    plt.close(figure)


def plot_sensitivity(results: Path, output: Path) -> None:
    gamma = pd.concat(
        [pd.read_csv(results / f"gamma_{str(value).replace('.', 'p')}_evaluation.csv")
         for value in GAMMAS], ignore_index=True
    )
    gamma_summary = seed_summary(gamma, ["variant", "gamma"])
    length = pd.read_csv(results / "training_length_evaluation.csv")
    length_summary = seed_summary(length, ["variant", "checkpoint_episode"])
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for variant in VARIANTS:
        part = gamma_summary[gamma_summary.variant == variant].sort_values("gamma")
        axes[0].errorbar(part.gamma, part["mean"],
                         yerr=np.vstack((part["mean"] - part.ci_low,
                                        part.ci_high - part["mean"])),
                         marker="o", ms=4, lw=1.2, capsize=2,
                         color=COLORS[variant], label=LABELS[variant])
        part = length_summary[length_summary.variant == variant].sort_values(
            "checkpoint_episode"
        )
        axes[1].errorbar(part.checkpoint_episode, part["mean"],
                         yerr=np.vstack((part["mean"] - part.ci_low,
                                        part.ci_high - part["mean"])),
                         marker="o", ms=4, lw=1.2, capsize=2,
                         color=COLORS[variant], label=LABELS[variant])
    axes[0].axvline(.97, color="#666666", ls="--", lw=.8)
    axes[0].set_xlabel("Discount factor, $\\gamma$")
    axes[1].set_xlabel("Training episodes")
    for label, axis in zip(("a", "b"), axes):
        axis.set_ylabel("Held-out episode cost")
        axis.grid(axis="y", color="#D9D9D9", lw=.55)
        axis.text(-.10, 1.03, label, transform=axis.transAxes, fontweight="bold")
        axis.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="lower center", frameon=False,
                  fontsize=8, bbox_to_anchor=(.5, -.02))
    figure.tight_layout(rect=(0, .10, 1, 1), pad=.8)
    save_figure_bundle(figure, output)
    plt.close(figure)


def latex_escape(value: str) -> str:
    return value.replace("&", "\\&")


def write_runtime_table(results: Path, output: Path) -> None:
    gamma = pd.read_csv(results / "gamma_0p97_evaluation.csv")
    baseline = pd.read_csv(results / "baseline_evaluation.csv")
    baseline = baseline[(baseline.gamma == .97) & baseline.policy.isin(["mpc4", "greedy8"])]
    baseline = baseline.rename(columns={"policy": "variant"})
    cost = seed_summary(pd.concat([gamma, baseline], ignore_index=True), ["variant"])
    runtime = pd.read_csv(results / "runtime_benchmark_summary.csv")
    merged = cost.merge(runtime, left_on="variant", right_on="policy")
    order = list(VARIANTS) + ["mpc4", "greedy8"]
    merged["order"] = merged.variant.map({name: i for i, name in enumerate(order)})
    merged = merged.sort_values("order")
    lines = [
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Policy & Episode cost, mean $\\pm$ SD & Mean (ms) & Median (ms) & P95 (ms) \\\\",
        "\\midrule",
    ]
    for row in merged.itertuples():
        lines.append(
            f"{latex_escape(LABELS[row.variant])} & {row.mean:.2f} $\\pm$ {row.sd:.2f} & "
            f"{row.decision_ms_mean:.3f} & {row.decision_ms_median:.3f} & "
            f"{row.decision_ms_p95:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(results: Path, output: Path) -> None:
    gamma = pd.read_csv(results / "gamma_sensitivity_summary.csv")
    length = pd.read_csv(results / "training_length_summary.csv")
    runtime = pd.read_csv(results / "runtime_benchmark_summary.csv")
    files = [
        "gamma_sensitivity_inference.csv", "dqn_vs_mpc_gamma_inference.csv",
        "dqn_vs_greedy8_inference.csv", "training_length_inference.csv",
    ]
    paragraphs = [
        "# Revision-round-2 result audit", "",
        "All summaries use 20 model-seed blocks after averaging 20 shared held-out "
        "traces within each seed. Bootstrap intervals use 10,000 seed-block resamples. "
        "Each added comparison family receives its own Holm correction.", "",
        f"Gamma cells: {len(gamma)}; training-length cells: {len(length)}; "
        f"runtime policies: {len(runtime)}.", "",
        "## Inference files", "",
    ]
    paragraphs.extend(f"- `{name}`" for name in files)
    output.write_text("\n".join(paragraphs) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manuscript-dir", type=Path, required=True)
    args = parser.parse_args()
    figures = args.manuscript_dir / "figures"
    generated = args.manuscript_dir / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(args.results, figures / "fig_revision_learning_curves.pdf")
    plot_sensitivity(args.results, figures / "fig_revision_sensitivity.pdf")
    write_runtime_table(args.results, generated / "table_revision_runtime.tex")
    write_report(args.results, args.results / "revision_round2_report.md")


if __name__ == "__main__":
    main()
