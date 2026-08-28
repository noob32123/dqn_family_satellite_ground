"""Generate LaTeX tables for the second-round reviewer analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from .dqn_mpc_family_inference import compute_family_inference, write_latex_tables
from .reviewer_experiments import DEFAULT_RESULTS


ROOT = Path(__file__).resolve().parent
POLICY_LABELS = {
    "standard_dqn": "Standard DQN",
    "full_action_q_dqn": "Full-action Q auxiliary",
    "immediate_advantage_dqn": "Immediate advantage auxiliary",
    "centered_full_action_dqn": "Centered full-action",
    "double_dqn": "Double DQN",
    "double_centered_full_action_dqn": "Double + centered full-action",
}
SCENARIO_LABELS = {
    "nominal": "Nominal",
    "burst": "Burst",
    "link_limited": "Link-limited",
    "energy_limited": "Energy-limited",
    "thermal_stress": "Thermal-stress",
}


def _latex_macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"


def _write_macros(results: Path, output: Path) -> None:
    inference = pd.read_csv(results / "extended_ablation_inference.csv")
    family = pd.read_csv(results / "extended_ablation_summary.csv")
    primary = pd.read_csv(results / "seven_regimes_summary.csv")
    planner = pd.read_csv(results / "planner_depth_summary.csv").set_index(
        "planner_horizon"
    )

    def contrast(comparison: str) -> pd.Series:
        return inference[
            inference.scenario.eq("nominal")
            & inference.comparison.eq(comparison)
        ].iloc[0]

    fullq = contrast("full_action_q_dqn - standard_dqn")
    centered = contrast("centered_full_action_dqn - full_action_q_dqn")
    immediate = contrast("immediate_advantage_dqn - standard_dqn")
    bootstrap = contrast("centered_full_action_dqn - immediate_advantage_dqn")
    double = contrast("double_centered_full_action_dqn - double_dqn")
    values = {
        "FullQDifference": f"{fullq.mean_difference:.2f}",
        "FullQHolmP": f"{fullq.holm_adjusted_p:.4g}",
        "CenteredDifference": f"{centered.mean_difference:.2f}",
        "CenteredCILow": f"{centered.ci95_low:.2f}",
        "CenteredCIHigh": f"{centered.ci95_high:.2f}",
        "CenteredHolmP": f"{centered.holm_adjusted_p:.4g}",
        "ImmediateDifference": f"{immediate.mean_difference:.2f}",
        "ImmediateHolmP": f"{immediate.holm_adjusted_p:.4g}",
        "BootstrapDifference": f"{bootstrap.mean_difference:.2f}",
        "BootstrapCILow": f"{bootstrap.ci95_low:.2f}",
        "BootstrapCIHigh": f"{bootstrap.ci95_high:.2f}",
        "BootstrapHolmP": f"{bootstrap.holm_adjusted_p:.4g}",
        "DoubleDifference": f"{double.mean_difference:.2f}",
        "DoubleCILow": f"{double.ci95_low:.2f}",
        "DoubleCIHigh": f"{double.ci95_high:.2f}",
        "DoubleHolmP": f"{double.holm_adjusted_p:.4g}",
    }
    for macro_name, comparison_name in (
        ("FullQSupportedCount", "full_action_q_dqn - standard_dqn"),
        ("CenteredFASupportedCount", "centered_full_action_dqn - standard_dqn"),
        ("CenteredSupportedCount", "centered_full_action_dqn - full_action_q_dqn"),
        (
            "ImmediateSupportedCount",
            "immediate_advantage_dqn - standard_dqn",
        ),
        (
            "BootstrapSupportedCount",
            "centered_full_action_dqn - immediate_advantage_dqn",
        ),
        ("DoubleSupportedCount", "double_centered_full_action_dqn - double_dqn"),
    ):
        values[macro_name] = str(
            int(inference[inference.comparison.eq(comparison_name)].supported.sum())
        )
    nominal_family = family[family.scenario.eq("nominal")]
    values["DQNFamilyNominalMinCost"] = (
        f"{nominal_family.total_cost_mean.min():.2f}"
    )
    values["DQNFamilyNominalMaxCost"] = (
        f"{nominal_family.total_cost_mean.max():.2f}"
    )
    mpc_by_scenario = (
        primary[primary.policy.eq("mpc_h4")]
        .set_index("scenario")
        .total_cost_mean
    )
    values["DQNFamilyBelowMPCCount"] = str(
        int(
            sum(
                row.total_cost_mean < mpc_by_scenario.loc[row.scenario]
                for row in family.itertuples()
            )
        )
    )
    horizon_names = {1: "One", 2: "Two", 4: "Four", 6: "Six"}
    for horizon in (1, 2, 4, 6):
        row = planner.loc[horizon]
        suffix = horizon_names[horizon]
        values[f"PlannerCostH{suffix}"] = f"{row.total_cost_mean:.2f}"
        values[f"PlannerTimeH{suffix}"] = f"{row.planning_ms_mean:.2f}"
    (output / "extended_macros.tex").write_text(
        "\n".join(_latex_macro(name, value) for name, value in values.items())
        + "\n",
        encoding="utf-8",
    )


def _write_ablation(results: Path, output: Path) -> None:
    summary = pd.read_csv(results / "extended_ablation_summary.csv")
    inference = pd.read_csv(results / "extended_ablation_inference.csv")
    primary = pd.read_csv(results / "seven_regimes_paired_bootstrap_complete.csv")
    nominal = summary[summary.scenario == "nominal"].set_index("policy")
    comparisons = {
        "full_action_q_dqn": "full_action_q_dqn - standard_dqn",
        "immediate_advantage_dqn": "immediate_advantage_dqn - standard_dqn",
        "centered_full_action_dqn": "centered_full_action_dqn - standard_dqn",
        "double_centered_full_action_dqn": "double_centered_full_action_dqn - double_dqn",
    }
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Training objective & Mean cost $\pm$ SD & Matched reference & Paired difference [95\% interval] & Holm $p$ & Supported \\",
        r"\midrule",
    ]
    for policy in POLICY_LABELS:
        row = nominal.loc[policy]
        if policy in comparisons:
            inf = inference[
                (inference.scenario == "nominal")
                & (inference.comparison == comparisons[policy])
            ].iloc[0]
            if policy == "centered_full_action_dqn":
                inf = primary[
                    primary.scenario.eq("nominal")
                    & primary.comparison.eq("centered_full_action_dqn - standard_dqn")
                ].iloc[0]
            reference = "Double DQN" if policy == "double_centered_full_action_dqn" else "Standard DQN"
            difference = (
                f"{inf.mean_difference:.2f} "
                f"[{inf.ci95_low:.2f}, {inf.ci95_high:.2f}]"
            )
            holm = f"{inf.holm_adjusted_p:.4g}"
            supported = "Yes" if bool(inf.supported) else "No"
        else:
            reference = "--"
            difference = "--"
            holm = "--"
            supported = "--"
        lines.append(
            f"{POLICY_LABELS[policy]} & {row.total_cost_mean:.2f} $\\pm$ "
            f"{row.total_cost_sd:.2f} & {reference} & {difference} & {holm} & "
            f"{supported} \\\\" 
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_extended_ablation.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_dqn_family_performance(results: Path, output: Path) -> None:
    summary = pd.read_csv(results / "extended_ablation_summary.csv")
    scenarios = list(SCENARIO_LABELS)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Training objective & "
        + " & ".join(SCENARIO_LABELS[item] for item in scenarios)
        + r" \\",
        r"\midrule",
    ]
    for policy, label in POLICY_LABELS.items():
        subset = summary[summary.policy.eq(policy)].set_index("scenario")
        cells = [
            f"{subset.loc[item, 'total_cost_mean']:.2f} $\\pm$ "
            f"{subset.loc[item, 'total_cost_sd']:.2f}"
            for item in scenarios
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_dqn_family_performance.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_planner_depth(results: Path, output: Path) -> None:
    summary = pd.read_csv(results / "planner_depth_summary.csv")
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Planning horizon & Enumerated sequences & Mean cost & Time per decision (ms) & Held-out traces \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{int(row.planner_horizon)} & {3 ** int(row.planner_horizon):,} & "
            f"{row.total_cost_mean:.2f} & {row.planning_ms_mean:.2f} & "
            f"{int(row.n_traces)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_planner_depth.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_unpaired(results: Path, output: Path) -> None:
    inference = pd.read_csv(results / "primary_unpaired_seed_bootstrap.csv")
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Regime & Mean difference & Unpaired 95\% interval & Excludes zero \\",
        r"\midrule",
    ]
    for scenario in SCENARIO_LABELS:
        row = inference[inference.scenario == scenario].iloc[0]
        lines.append(
            f"{SCENARIO_LABELS[scenario]} & {row.mean_difference:.2f} & "
            f"[{row.ci95_low:.2f}, {row.ci95_high:.2f}] & "
            f"{'Yes' if bool(row.excludes_zero) else 'No'} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_unpaired_inference.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_training_accounting(results: Path, output: Path) -> None:
    manifest = pd.read_csv(results / "model_manifest.csv")
    summary = manifest.groupby("variant", sort=False).agg(
        wall_mean=("training_wall_time_s", "mean"),
        wall_sd=("training_wall_time_s", "std"),
        real_interactions=("training_steps", "first"),
        optimizer_updates=("optimizer_updates", "first"),
        preview_evaluations=("preview_action_evaluations", "first"),
        n_models=("model_seed", "nunique"),
    )
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Training objective & Wall time per model (s) & Real interactions & Updates & Modeled action outcomes & Models \\",
        r"\midrule",
    ]
    for policy in POLICY_LABELS:
        if policy not in summary.index:
            continue
        row = summary.loc[policy]
        wall_sd = 0.0 if pd.isna(row.wall_sd) else float(row.wall_sd)
        lines.append(
            f"{POLICY_LABELS[policy]} & {row.wall_mean:.1f} $\\pm$ {wall_sd:.1f} & "
            f"{int(row.real_interactions):,} & {int(row.optimizer_updates):,} & "
            f"{int(row.preview_evaluations):,} & "
            f"{int(row.n_models)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_training_accounting.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _augment_validation(results: Path, output: Path) -> None:
    validation_path = results / "extended_revision_validation.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    result_patterns = (
        "extended_ablation_*.csv",
        "dqn_vs_mpc_*.csv",
        "planner_depth_*.csv",
        "primary_unpaired_seed_bootstrap.csv",
        "extended_revision_metadata.json",
        "model_manifest.csv",
    )
    result_files = sorted(
        {path for pattern in result_patterns for path in results.glob(pattern)}
    )
    source_files = (
        ROOT / "agent.py",
        ROOT / "environment.py",
        ROOT / "experiment.py",
        ROOT / "planner.py",
        ROOT / "reviewer_experiments.py",
        ROOT / "reviewer_reporting.py",
        ROOT / "extended_revision.py",
        ROOT / "extended_reporting.py",
        ROOT / "dqn_mpc_family_inference.py",
        ROOT / "transition_audit.py",
        ROOT / "plot_reviewer_results_matlab.m",
        ROOT / "plot_dqn_family_summary.py",
        ROOT / "requirements.txt",
        ROOT.parent / "dqn_family_satellite_ground_manuscript.pdf",

    )
    table_files = sorted(output.glob("*.tex"))
    payload["artifacts_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result_files
    }
    payload["source_sha256"] = {
        str(path.relative_to(ROOT.parent)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_files
    }
    payload["generated_tables_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in table_files
    }
    validation_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=ROOT / "tables" / "reviewer_revision_state_complete",
    )
    parser.add_argument("--paper", type=Path, default=ROOT.parent / "reproduced_outputs")
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    _write_ablation(args.results, args.tables)
    _write_dqn_family_performance(args.results, args.tables)
    _write_planner_depth(args.results, args.tables)
    _write_unpaired(args.results, args.tables)
    _write_training_accounting(args.results, args.tables)
    _write_macros(args.results, args.tables)
    cells, maxima, seeds = compute_family_inference(args.results)
    cells.to_csv(args.results / "dqn_vs_mpc_inference.csv", index=False)
    maxima.to_csv(args.results / "dqn_vs_mpc_family_inference.csv", index=False)
    seeds.to_csv(args.results / "dqn_vs_mpc_seed_differences.csv", index=False)
    write_latex_tables(cells, maxima, args.tables)
    generated = args.paper / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for name in (
        "table_extended_ablation.tex",
        "table_dqn_family_performance.tex",
        "table_planner_depth.tex",
        "table_unpaired_inference.tex",
        "table_training_accounting.tex",
        "table_dqn_vs_mpc_family_inference.tex",
        "dqn_vs_mpc_macros.tex",
        "extended_macros.tex",
    ):
        shutil.copy2(args.tables / name, generated / name)
    _augment_validation(args.results, args.tables)


if __name__ == "__main__":
    main()
