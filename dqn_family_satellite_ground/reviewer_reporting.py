"""Generate reviewer-revision manuscript tables from locked CSV outputs.

This module formats tables only. All quantitative manuscript figures are
drawn by MATLAB in ``plot_reviewer_results_matlab.m``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from .experiment import FULL_COUPLING, SCENARIOS
from .reviewer_experiments import DEFAULT_RESULTS, paired_bootstrap


ROOT = Path(__file__).resolve().parent
POLICIES = (
    "immediate_argmin",
    "mpc_h4",
    "contextual_bandit",
    "threshold",
    "standard_dqn",
    "centered_full_action_dqn",
)
POLICY_LABELS = {
    "immediate_argmin": "Argmin",
    "mpc_h4": "MPC-4",
    "contextual_bandit": "Bandit",
    "threshold": "Threshold",
    "standard_dqn": "DQN",
    "centered_full_action_dqn": "Centered FA",
}
SCENARIO_LABELS = {
    "static": "Static ($c=0$)",
    "reduced_coupling": "Reduced ($c=0.5$)",
    "nominal": "Nominal",
    "burst": "Burst",
    "link_limited": "Link-limited",
    "energy_limited": "Energy-limited",
    "thermal_stress": "Thermal-stress",
}
PROFILE_LABELS = {
    "reference": "Reference",
    "compute_minus20": r"Compute demand $-20\%$",
    "compute_plus20": r"Compute demand $+20\%$",
    "data_minus20": r"Input data $-20\%$",
    "data_plus20": r"Input data $+20\%$",
    "energy_plus20": r"Energy use $+20\%$",
    "heat_plus20": r"Heat load $+20\%$",
    "link_minus20": r"Link capacity $-20\%$",
}


def _metric(summary: pd.DataFrame, scenario: str, policy: str, column: str) -> float:
    row = summary[(summary.scenario == scenario) & (summary.policy == policy)]
    if len(row) != 1:
        raise ValueError(f"expected one row for {scenario}/{policy}, found {len(row)}")
    return float(row.iloc[0][column])


def _write_main(summary: pd.DataFrame, output: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Regime & Argmin & MPC-4 & Bandit & Threshold & DQN & Centered FA \\",
        r"\midrule",
    ]
    for scenario, _, _ in SCENARIOS:
        values = {p: _metric(summary, scenario, p, "total_cost_mean") for p in POLICIES}
        cells = []
        for policy in POLICIES:
            mean = values[policy]
            sd = _metric(summary, scenario, policy, "total_cost_sd")
            cell = f"{mean:.2f} $\\pm$ {sd:.2f}"
            cells.append(cell)
        lines.append(SCENARIO_LABELS[scenario] + " & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_reviewer_main.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_sensitivity(
    sensitivity: pd.DataFrame, primary: pd.DataFrame, output: Path
) -> None:
    selected = sensitivity[sensitivity.comparison == "centered_full_action_dqn - standard_dqn"]
    primary_nominal = primary[
        primary.comparison.eq("centered_full_action_dqn - standard_dqn")
        & primary.scenario.eq("nominal")
    ].iloc[0]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Parameter profile & Mean difference & 95\% interval & Improvement & Holm $p$ \\",
        r"\midrule",
    ]
    for profile in PROFILE_LABELS:
        row = selected[selected.profile == profile].iloc[0]
        ci_low = primary_nominal.ci95_low if profile == "reference" else row.ci95_low
        ci_high = primary_nominal.ci95_high if profile == "reference" else row.ci95_high
        lines.append(
            f"{PROFILE_LABELS[profile]} & {row.mean_difference:.2f} & "
            f"[{ci_low:.2f}, {ci_high:.2f}] & "
            f"{row.relative_improvement_percent:.2f}\\% & "
            f"{row.holm_adjusted_p:.4g} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_parameters(output: Path) -> None:
    rows = (
        ("Raw task data", "8--64 Mbit nominal", "Stress interval reaches 92.8 Mbit before sensitivity scaling"),
        ("Arithmetic workload", "0.25--4.0 GFLOP nominal", "Stress interval reaches 5.8 GFLOP"),
        ("Result / raw-data ratio", "1--5\\%", "Bounded compression assumption"),
        ("Feature / raw-data ratio", "8--30\\%", "Bounded collaborative-processing assumption"),
        ("Preprocessing fraction", "20--45\\%", "Deterministic function of workload quantile"),
        ("Onboard / ground rate", "4 / 80 GFLOP s$^{-1}$", "Fixed simulator hardware envelope"),
        ("Compute power", "4--20 W", "Bracketed by NASA SmallSat avionics examples"),
        ("Effective RF return rate", "2.2--220 Mbit s$^{-1}$", "NASA SmallSat ground-system examples"),
        ("Transmit power", "6 W", "Fixed simulator assumption"),
        ("Burst multiplier", "$1.45\\times$", "Contiguous prespecified stressor"),
    )
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Quantity & Implemented range & Basis \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_parameter_provenance.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_state_dictionary(output: Path) -> None:
    rows = (
        ("$x_1$", "Onboard compute heat", "J", "100"),
        ("$x_2$", "Total workload", "GFLOP", "4.8"),
        ("$x_3$", "Onboard compute time", "ms", "1400"),
        ("$x_4$", "Result transmit time at 50 Mbit s$^{-1}$", "ms", "100"),
        ("$x_5$", "Squared co-location count", "count$^2$", "2401"),
        ("$x_6$", "Onboard-result data", "Mbit", "4"),
        ("$x_7$", "Full-offload transmit heat", "J", "100"),
        ("$x_8$", "Ground compute time", "ms", "100"),
        ("$x_9$", "Raw input data", "Mbit", "76.8"),
        ("$x_{10}$", "Preprocessing heat", "J", "100"),
        ("$x_{11}$", "Feature-transmit heat", "J", "100"),
        ("$x_{12}$", "Onboard preprocessing work", "GFLOP", "2.2"),
        ("$x_{13}$", "Remaining ground compute time", "ms", "100"),
        ("$x_{14}$", "Total workload descriptor", "GFLOP", "4.8"),
        ("$x_{15}$", "Collaborative feature data", "Mbit", "32"),
    )
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Coordinate & Descriptor & Unit & Normalization scale $s_k$ \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_state_dictionary.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_primary_inference(comparison: pd.DataFrame, output: Path) -> None:
    full_names = [row[0] for row in FULL_COUPLING]
    selected = comparison[
        (comparison.comparison == "centered_full_action_dqn - standard_dqn")
        & comparison.scenario.isin(full_names)
    ].set_index("scenario")
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Regime & Mean difference & 95\% interval & Negative seeds & Holm $p$ & LOO range \\",
        r"\midrule",
    ]
    for scenario in full_names:
        row = selected.loc[scenario]
        lines.append(
            f"{SCENARIO_LABELS[scenario]} & {row.mean_difference:.2f} & "
            f"[{row.ci95_low:.2f}, {row.ci95_high:.2f}] & "
            f"{int(row.negative_seed_pairs)}/{int(row.n_pairs)} & "
            f"{row.holm_adjusted_p:.4g} & "
            f"[{row.loo_mean_min:.2f}, {row.loo_mean_max:.2f}] \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_primary_inference.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_engineering_outcomes(summary: pd.DataFrame, output: Path) -> None:
    scenarios = (
        "nominal", "burst", "link_limited", "energy_limited", "thermal_stress"
    )
    delta_lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Regime & $\Delta$ immediate & $\Delta$ penalty & $\Delta$ energy & $\Delta$ latency (s) & $\Delta$ data (Mbit) \\",
        r"\midrule",
    ]
    exposure_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Regime & Low-energy steps DQN/MPC-4 & Min. energy DQN/MPC-4 & Max. heat DQN/MPC-4 \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        def delta(column: str) -> float:
            return _metric(summary, scenario, "standard_dqn", column) - _metric(
                summary, scenario, "mpc_h4", column
            )
        energy_dqn = _metric(summary, scenario, "standard_dqn", "energy_violation_mean")
        energy_mpc = _metric(summary, scenario, "mpc_h4", "energy_violation_mean")
        minimum_energy_dqn = _metric(summary, scenario, "standard_dqn", "minimum_energy_mean")
        minimum_energy_mpc = _metric(summary, scenario, "mpc_h4", "minimum_energy_mean")
        heat_dqn = _metric(summary, scenario, "standard_dqn", "maximum_heat_mean")
        heat_mpc = _metric(summary, scenario, "mpc_h4", "maximum_heat_mean")
        delta_lines.append(
            f"{SCENARIO_LABELS[scenario]} & {delta('immediate_cost_mean'):.2f} & "
            f"{delta('penalty_mean'):.2f} & {delta('energy_use_mean'):.3f} & "
            f"{delta('latency_ms_mean') / 1000.0:.2f} & "
            f"{delta('transmitted_mbit_mean'):.1f} \\\\"
        )
        exposure_lines.append(
            f"{SCENARIO_LABELS[scenario]} & {energy_dqn:.1f}/{energy_mpc:.1f} & "
            f"{minimum_energy_dqn:.2f}/{minimum_energy_mpc:.2f} & "
            f"{heat_dqn:.3f}/{heat_mpc:.3f} \\\\"
        )
    delta_lines.extend((r"\bottomrule", r"\end{tabular}"))
    exposure_lines.extend((r"\bottomrule", r"\end{tabular}"))
    lines = delta_lines + [r"\par\medskip"] + exposure_lines
    (output / "table_engineering_outcomes.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_preview_mismatch(
    inference: pd.DataFrame, primary: pd.DataFrame, output: Path
) -> None:
    labels = {
        "centered_full_action_dqn_exact - standard_dqn": "Simulator-exact preview",
        "centered_full_action_dqn_preview_0p85 - standard_dqn": "Consequences scaled to 0.85",
        "centered_full_action_dqn_preview_1p15 - standard_dqn": "Consequences scaled to 1.15",
    }
    scenarios = [row[0] for row in FULL_COUPLING]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Training preview & Regime & Mean difference & 95\% interval & Negative seeds & Holm $p$ & Supported \\",
        r"\midrule",
    ]
    for comparison, label in labels.items():
        if comparison == "centered_full_action_dqn_exact - standard_dqn":
            subset = primary[
                primary.comparison.eq("centered_full_action_dqn - standard_dqn")
            ].set_index("scenario")
        else:
            subset = inference[inference.comparison == comparison].set_index("scenario")
        for index, scenario in enumerate(scenarios):
            row = subset.loc[scenario]
            first = label if index == 0 else ""
            lines.append(
                f"{first} & {SCENARIO_LABELS[scenario]} & {row.mean_difference:.2f} & "
                f"[{row.ci95_low:.2f}, {row.ci95_high:.2f}] & "
                f"{int(row.negative_seed_pairs)}/{int(row.n_pairs)} & "
                f"{row.holm_adjusted_p:.4g} & "
                f"{'Yes' if bool(row.supported) else 'No'} \\\\"
            )
        if comparison != list(labels)[-1]:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_preview_mismatch.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_macros(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    sensitivity: pd.DataFrame,
    training: pd.DataFrame,
    mismatch: pd.DataFrame | None,
    validity: dict,
    planner_depth: pd.DataFrame | None,
    output: Path,
) -> None:
    full_names = [row[0] for row in FULL_COUPLING]
    centered_full_action_dqn = comparison[
        (comparison.comparison == "centered_full_action_dqn - standard_dqn")
        & comparison.scenario.isin(full_names)
    ]
    sens = sensitivity[sensitivity.comparison == "centered_full_action_dqn - standard_dqn"]
    if planner_depth is not None:
        planning_per_step = float(
            planner_depth.loc[
                planner_depth.planner_horizon.eq(4), "planning_ms_mean"
            ].iloc[0]
        )
    else:
        planning_per_step = _metric(
            summary, "nominal", "mpc_h4", "planning_time_ms_mean"
        ) / 64.0
    late_start = max(0, int(training.episode.max()) - 99)
    late = training[
        training.variant.isin(("standard_dqn", "centered_full_action_dqn"))
        & (training.episode >= late_start)
    ]
    late_means = late.groupby("variant")[["cost", "energy_use", "latency_ms", "penalty"]].mean()
    dqn_late = late_means.loc["standard_dqn"]
    centered_full_action_late = late_means.loc["centered_full_action_dqn"]
    macros = {
        "PhysicalCenteredFAMinPct": f"{centered_full_action_dqn.relative_improvement_percent.min():.2f}\\%",
        "PhysicalCenteredFAMaxPct": f"{centered_full_action_dqn.relative_improvement_percent.max():.2f}\\%",
        "SensitivityMinPct": f"{sens.relative_improvement_percent.min():.2f}\\%",
        "SensitivityMaxPct": f"{sens.relative_improvement_percent.max():.2f}\\%",
        "PhysicalCorrDataWork": f"{validity['corr_raw_workload']:.3f}",
        "MPCNominalPlanningMs": f"{planning_per_step:.2f}",
        "MPCNominalCost": f"{_metric(summary, 'nominal', 'mpc_h4', 'total_cost_mean'):.2f}",
        "PhysicalNominalDQNCost": f"{_metric(summary, 'nominal', 'standard_dqn', 'total_cost_mean'):.2f}",
        "PhysicalNominalCenteredFACost": f"{_metric(summary, 'nominal', 'centered_full_action_dqn', 'total_cost_mean'):.2f}",
        "PhysicalAuditTasks": f"{int(validity['n_tasks']):,}",
        "IndependentModelSeeds": f"{int(summary.n_model_seeds.max())}",
        "PrimaryHolmPMax": f"{centered_full_action_dqn.holm_adjusted_p.max():.4g}",
        "PrimarySupportedCount": f"{int(centered_full_action_dqn.supported.sum())}",
        "PrimaryNegativeSeedsMin": f"{int(centered_full_action_dqn.negative_seed_pairs.min())}",
        "PrimaryNegativeSeedsMax": f"{int(centered_full_action_dqn.negative_seed_pairs.max())}",
        "LateCenteredFACost": f"{centered_full_action_late.cost:.2f}",
        "LateDQNCost": f"{dqn_late.cost:.2f}",
        "LateCenteredFAEnergy": f"{centered_full_action_late.energy_use:.2f}",
        "LateDQNEnergy": f"{dqn_late.energy_use:.2f}",
        "LateCenteredFALatency": f"{centered_full_action_late.latency_ms / 1000.0:.2f}",
        "LateDQNLatency": f"{dqn_late.latency_ms / 1000.0:.2f}",
        "LateCenteredFAPenalty": f"{centered_full_action_late.penalty:.2f}",
        "LateDQNPenalty": f"{dqn_late.penalty:.2f}",
    }
    if mismatch is not None:
        misspecified = mismatch[
            mismatch.comparison.isin(
                (
                    "centered_full_action_dqn_preview_0p85 - standard_dqn",
                    "centered_full_action_dqn_preview_1p15 - standard_dqn",
                )
            )
        ]
        macros["MismatchHolmPMax"] = f"{misspecified.holm_adjusted_p.max():.4g}"
        for scale, comparison_name in (
            ("Low", "centered_full_action_dqn_preview_0p85 - standard_dqn"),
            ("High", "centered_full_action_dqn_preview_1p15 - standard_dqn"),
        ):
            subset = mismatch[mismatch.comparison.eq(comparison_name)]
            macros[f"Mismatch{scale}HolmPMax"] = (
                f"{subset.holm_adjusted_p.max():.4g}"
            )
            macros[f"Mismatch{scale}SupportedCount"] = (
                f"{int(subset.supported.sum())}"
            )
    lines = [f"\\newcommand{{\\{key}}}{{{value}\\xspace}}" for key, value in macros.items()]
    (output / "reviewer_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--tables", type=Path, default=ROOT / "tables" / "reviewer_revision_state_complete"
    )
    parser.add_argument("--paper", type=Path, default=ROOT.parent / "reproduced_outputs")
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.results / "seven_regimes_raw.csv")
    summary = pd.read_csv(args.results / "seven_regimes_summary.csv")
    comparison = paired_bootstrap(
        raw,
        (
            ("centered_full_action_dqn", "standard_dqn"),
            ("centered_full_action_dqn", "mpc_h4"),
            ("mpc_h4", "immediate_argmin"),
        ),
        inference_family=tuple(item[0] for item in FULL_COUPLING),
    )
    comparison.to_csv(args.results / "seven_regimes_paired_bootstrap_complete.csv", index=False)
    sensitivity = pd.read_csv(args.results / "sensitivity_paired_bootstrap.csv")
    training = pd.read_csv(args.results / "training_curves.csv")
    validity = pd.read_json(args.results / "parameter_validity.json", typ="series").to_dict()

    _write_main(summary, args.tables)
    _write_sensitivity(sensitivity, comparison, args.tables)
    _write_parameters(args.tables)
    _write_state_dictionary(args.tables)
    _write_primary_inference(comparison, args.tables)
    _write_engineering_outcomes(summary, args.tables)
    mismatch_path = args.results / "preview_mismatch_inference.csv"
    mismatch = pd.read_csv(mismatch_path) if mismatch_path.exists() else None
    planner_path = args.results / "planner_depth_summary.csv"
    planner_depth = pd.read_csv(planner_path) if planner_path.exists() else None
    if mismatch_path.exists():
        _write_preview_mismatch(mismatch, comparison, args.tables)
    _write_macros(
        summary, comparison, sensitivity, training, mismatch, validity,
        planner_depth, args.tables
    )

    generated = args.paper / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for source in args.tables.glob("*.tex"):
        shutil.copy2(source, generated / source.name)


if __name__ == "__main__":
    main()
