"""Direct paired inference for the six-objective DQN family versus MPC-4.

The independent unit is the model-seed block.  Each block first averages the
20 held-out traces shared by a DQN objective and MPC-4.  A single joint
bootstrap then resamples the 20 seed blocks across all 30 objective--regime
contrasts, preserving their correlation for family-wide uncertainty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .reviewer_experiments import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_RESULTS,
    _exact_two_sided_sign_p,
    _holm_adjust,
)


DQN_POLICIES = (
    "standard_dqn",
    "full_action_q_dqn",
    "immediate_advantage_dqn",
    "centered_full_action_dqn",
    "double_dqn",
    "double_centered_full_action_dqn",
)
SCENARIOS = (
    "nominal",
    "burst",
    "link_limited",
    "energy_limited",
    "thermal_stress",
)
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


def compute_family_inference(
    results: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cell-level, maximum-statistic, and seed-level results."""
    dqn_raw = pd.read_csv(results / "extended_ablation_raw.csv")
    primary_raw = pd.read_csv(results / "seven_regimes_raw.csv")
    dqn_raw = dqn_raw[
        dqn_raw.policy.isin(DQN_POLICIES) & dqn_raw.scenario.isin(SCENARIOS)
    ].copy()
    mpc_raw = primary_raw[
        primary_raw.policy.eq("mpc_h4") & primary_raw.scenario.isin(SCENARIOS)
    ][["scenario", "model_seed", "trace_seed", "total_cost"]].rename(
        columns={"total_cost": "mpc_total_cost"}
    )
    paired = dqn_raw.merge(
        mpc_raw,
        on=["scenario", "model_seed", "trace_seed"],
        how="left",
        validate="many_to_one",
    )
    if paired.mpc_total_cost.isna().any() or len(paired) != len(dqn_raw):
        raise AssertionError("DQN and MPC-4 trace namespaces are not fully paired")
    paired["difference"] = paired.total_cost - paired.mpc_total_cost
    seed_level = (
        paired.groupby(["scenario", "policy", "model_seed"], as_index=False)
        .agg(
            mean_difference=("difference", "mean"),
            dqn_total_cost=("total_cost", "mean"),
            mpc_total_cost=("mpc_total_cost", "mean"),
            n_traces=("trace_seed", "nunique"),
        )
    )
    if not seed_level.n_traces.eq(20).all():
        raise AssertionError("each seed block must contain exactly 20 held-out traces")

    ordered_columns = pd.MultiIndex.from_product(
        [SCENARIOS, DQN_POLICIES], names=["scenario", "policy"]
    )
    matrix_frame = seed_level.pivot(
        index="model_seed", columns=["scenario", "policy"], values="mean_difference"
    ).reindex(columns=ordered_columns)
    if matrix_frame.shape != (20, 30) or matrix_frame.isna().any().any():
        raise AssertionError(
            f"expected a complete 20-by-30 seed matrix, found {matrix_frame.shape}"
        )
    matrix = matrix_frame.to_numpy(dtype=float)
    observed = matrix.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, matrix.shape[0], size=(BOOTSTRAP_RESAMPLES, matrix.shape[0])
    )
    bootstrap_means = matrix[indices].mean(axis=1)
    global_deviation_quantile = float(
        np.quantile((bootstrap_means - observed).max(axis=1), 0.95)
    )

    sign_p = np.array(
        [_exact_two_sided_sign_p(matrix[:, index]) for index in range(matrix.shape[1])]
    )
    holm_p = _holm_adjust(sign_p)
    rows: list[dict] = []
    for index, (scenario, policy) in enumerate(ordered_columns):
        difference = matrix[:, index]
        ci_low, ci_high = np.quantile(bootstrap_means[:, index], (0.025, 0.975))
        mpc_mean = float(
            seed_level.loc[seed_level.scenario.eq(scenario), "mpc_total_cost"].mean()
        )
        rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "n_seed_blocks": len(difference),
                "traces_per_seed": 20,
                "mean_difference": float(observed[index]),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "familywise_95_upper_bound": float(
                    observed[index] + global_deviation_quantile
                ),
                "relative_improvement_percent": float(
                    -100.0 * observed[index] / mpc_mean
                ),
                "negative_seed_blocks": int((difference < 0).sum()),
                "positive_seed_blocks": int((difference > 0).sum()),
                "exact_sign_p": float(sign_p[index]),
                "holm_adjusted_p_30": float(holm_p[index]),
                "individual_ci_below_zero": bool(ci_high < 0),
                "holm_supported": bool(
                    observed[index] < 0 and ci_high < 0 and holm_p[index] < 0.05
                ),
            }
        )
    cell_level = pd.DataFrame(rows)

    maximum_rows: list[dict] = []
    scopes = [(scenario, [
        index for index, column in enumerate(ordered_columns) if column[0] == scenario
    ]) for scenario in SCENARIOS]
    scopes.append(("all_fully_coupled", list(range(len(ordered_columns)))))
    for scope, columns in scopes:
        scope_observed = observed[columns]
        scope_bootstrap = bootstrap_means[:, columns]
        worst_local_index = int(np.argmax(scope_observed))
        worst_column = columns[worst_local_index]
        max_bootstrap = scope_bootstrap.max(axis=1)
        max_ci_low, max_ci_high = np.quantile(max_bootstrap, (0.025, 0.975))
        deviation_quantile = float(
            np.quantile(
                (scope_bootstrap - scope_observed).max(axis=1),
                0.95,
            )
        )
        maximum_rows.append(
            {
                "scope": scope,
                "n_contrasts": len(columns),
                "least_favorable_scenario": ordered_columns[worst_column][0],
                "least_favorable_policy": ordered_columns[worst_column][1],
                "maximum_mean_difference": float(scope_observed[worst_local_index]),
                "max_statistic_ci95_low": float(max_ci_low),
                "max_statistic_ci95_high": float(max_ci_high),
                "simultaneous_95_upper_bound": float(
                    scope_observed.max() + deviation_quantile
                ),
                "all_individual_ci_below_zero": bool(
                    cell_level.iloc[columns].ci95_high.lt(0).all()
                ),
                "all_holm_supported": bool(
                    cell_level.iloc[columns].holm_supported.all()
                ),
            }
        )
    maximum_level = pd.DataFrame(maximum_rows)
    return cell_level, maximum_level, seed_level


def write_latex_tables(
    cell_level: pd.DataFrame, maximum_level: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Scope & Least-favorable DQN objective & Maximum $\Delta$ & 95\% interval for maximum & Simultaneous 95\% upper bound \\",
        r"\midrule",
    ]
    for row in maximum_level.itertuples(index=False):
        if row.scope == "all_fully_coupled":
            scope_label = "All five regimes"
            policy_label = (
                f"{POLICY_LABELS[row.least_favorable_policy]} "
                f"({SCENARIO_LABELS[row.least_favorable_scenario]})"
            )
        else:
            scope_label = SCENARIO_LABELS[row.scope]
            policy_label = POLICY_LABELS[row.least_favorable_policy]
        lines.append(
            f"{scope_label} & {policy_label} & {row.maximum_mean_difference:.2f} & "
            f"[{row.max_statistic_ci95_low:.2f}, {row.max_statistic_ci95_high:.2f}] & "
            f"{row.simultaneous_95_upper_bound:.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "table_dqn_vs_mpc_family_inference.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    overall = maximum_level[maximum_level.scope.eq("all_fully_coupled")].iloc[0]
    macros = {
        "DQNvsMPCContrastCount": str(len(cell_level)),
        "DQNvsMPCSupportedCount": str(int(cell_level.holm_supported.sum())),
        "DQNvsMPCGlobalWorstDifference": f"{overall.maximum_mean_difference:.2f}",
        "DQNvsMPCGlobalUpperBound": f"{overall.simultaneous_95_upper_bound:.2f}",
        "DQNvsMPCHolmPMax": f"{cell_level.holm_adjusted_p_30.max():.4g}",
        "DQNvsMPCNegativeSeedMin": str(int(cell_level.negative_seed_blocks.min())),
        "DQNvsMPCNegativeSeedMax": str(int(cell_level.negative_seed_blocks.max())),
    }
    macro_lines = [
        f"\\newcommand{{\\{name}}}{{{value}\\xspace}}"
        for name, value in macros.items()
    ]
    (output / "dqn_vs_mpc_macros.tex").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--tables",
        type=Path,
        default=Path(__file__).resolve().parent
        / "tables"
        / "reviewer_revision_state_complete",
    )
    args = parser.parse_args()
    cells, maxima, seeds = compute_family_inference(args.results)
    cells.to_csv(args.results / "dqn_vs_mpc_inference.csv", index=False)
    maxima.to_csv(args.results / "dqn_vs_mpc_family_inference.csv", index=False)
    seeds.to_csv(args.results / "dqn_vs_mpc_seed_differences.csv", index=False)
    write_latex_tables(cells, maxima, args.tables)


if __name__ == "__main__":
    main()
