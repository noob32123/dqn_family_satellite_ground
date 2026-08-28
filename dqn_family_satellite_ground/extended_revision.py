"""Mechanism ablations, Double-DQN transfer, unpaired inference, and planner depth.

This extension answers the reviewer requests while reusing the state-complete
standard-DQN and centered full-action checkpoints.  Every learned comparison uses the
same training budgets, model seeds, environment seeds, exploration stream, and
replay-sampling stream.  Planning-depth results are descriptive over 400 unique
held-out traces and are not treated as independently trained model replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .environment import SatelliteSchedulingEnv
from .experiment import AUXILIARY_LAMBDA, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from .planner import PlannerConfig, RecedingHorizonPlanner
from .reviewer_experiments import (
    CONFIRMATION_SEEDS,
    DEFAULT_RESULTS,
    FULL_COUPLING,
    TRACE_BASE,
    evaluate_policy,
    paired_bootstrap,
    paired_seed_differences,
    revision_env,
    seed_level,
    summarize,
    train_or_load_revision,
    write_model_manifest,
)


EXTENDED_VARIANTS = (
    "full_action_q_dqn",
    "immediate_advantage_dqn",
    "double_dqn",
    "double_centered_full_action_dqn",
)
ALL_LEARNED = (
    "standard_dqn",
    "full_action_q_dqn",
    "immediate_advantage_dqn",
    "centered_full_action_dqn",
    "double_dqn",
    "double_centered_full_action_dqn",
)
COMPARISONS = (
    ("full_action_q_dqn", "standard_dqn"),
    ("centered_full_action_dqn", "standard_dqn"),
    ("centered_full_action_dqn", "full_action_q_dqn"),
    ("immediate_advantage_dqn", "standard_dqn"),
    ("centered_full_action_dqn", "immediate_advantage_dqn"),
    ("double_centered_full_action_dqn", "double_dqn"),
)
PLANNER_HORIZONS = (1, 2, 4, 6)


def train_extended(
    results: Path,
    seeds: tuple[int, ...],
    episodes: int,
    horizon: int,
    device: torch.device,
    resume: bool,
) -> dict[tuple[str, int], object]:
    agents: dict[tuple[str, int], object] = {}
    for seed in seeds:
        for variant in ALL_LEARNED:
            print(f"training/loading {variant} seed={seed}", flush=True)
            agents[(variant, seed)] = train_or_load_revision(
                results, variant, seed, episodes, horizon, device, resume
            )
    return agents


def evaluate_ablations(
    results: Path,
    agents: dict[tuple[str, int], object],
    seeds: tuple[int, ...],
    horizon: int,
    test_traces: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed_index, seed in enumerate(seeds):
        traces = [TRACE_BASE + seed_index * 1000 + i for i in range(test_traces)]
        for label, scenario, coupling in FULL_COUPLING:
            config = revision_env(horizon, scenario=scenario, coupling=coupling)
            for policy in ALL_LEARNED:
                rows.extend(
                    evaluate_policy(policy, agents[(policy, seed)], seed, traces, config, label)
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(results / "extended_ablation_raw.csv", index=False)
    seed_level(raw).to_csv(results / "extended_ablation_seed_level.csv", index=False)
    summarize(raw).to_csv(results / "extended_ablation_summary.csv", index=False)
    inference = paired_bootstrap(
        raw,
        COMPARISONS,
        inference_family=tuple(item[0] for item in FULL_COUPLING),
    )
    inference.to_csv(results / "extended_ablation_inference.csv", index=False)
    paired_seed_differences(raw, COMPARISONS).to_csv(
        results / "extended_ablation_seed_differences.csv", index=False
    )
    return raw


def unpaired_seed_bootstrap(raw: pd.DataFrame) -> pd.DataFrame:
    by_seed = seed_level(raw)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    for label, _, _ in FULL_COUPLING:
        selected = by_seed[by_seed.scenario == label]
        treatment = selected[selected.policy == "centered_full_action_dqn"].total_cost.to_numpy()
        reference = selected[selected.policy == "standard_dqn"].total_cost.to_numpy()
        boot_t = rng.choice(
            treatment, size=(BOOTSTRAP_RESAMPLES, len(treatment)), replace=True
        ).mean(axis=1)
        boot_r = rng.choice(
            reference, size=(BOOTSTRAP_RESAMPLES, len(reference)), replace=True
        ).mean(axis=1)
        difference = boot_t - boot_r
        rows.append(
            {
                "scenario": label,
                "n_treatment": len(treatment),
                "n_reference": len(reference),
                "mean_difference": float(treatment.mean() - reference.mean()),
                "ci95_low": float(np.quantile(difference, 0.025)),
                "ci95_high": float(np.quantile(difference, 0.975)),
                "excludes_zero": bool(
                    np.quantile(difference, 0.975) < 0
                    or np.quantile(difference, 0.025) > 0
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_planner_depth(
    results: Path,
    seeds: tuple[int, ...],
    horizon: int,
    test_traces: int,
) -> pd.DataFrame:
    for planner_horizon in PLANNER_HORIZONS:
        warmup = SatelliteSchedulingEnv(revision_env(horizon))
        warmup.reset(seed=TRACE_BASE - planner_horizon)
        planner = RecedingHorizonPlanner(
            PlannerConfig(horizon=planner_horizon, gamma=0.97)
        )
        done = False
        while not done:
            action = planner.act(warmup)
            _, _, done, _ = warmup.step(action)
    rows: list[dict] = []
    for seed_index, _ in enumerate(seeds):
        traces = [TRACE_BASE + seed_index * 1000 + i for i in range(test_traces)]
        for trace_seed in traces:
            for planner_horizon in PLANNER_HORIZONS:
                env = SatelliteSchedulingEnv(revision_env(horizon))
                env.reset(seed=trace_seed)
                planner = RecedingHorizonPlanner(
                    PlannerConfig(horizon=planner_horizon, gamma=0.97)
                )
                total_cost = 0.0
                planning_ms = 0.0
                done = False
                while not done:
                    started = time.perf_counter()
                    action = planner.act(env)
                    planning_ms += 1000.0 * (time.perf_counter() - started)
                    _, _, done, info = env.step(action)
                    total_cost += info["total_cost"]
                rows.append(
                    {
                        "planner_horizon": planner_horizon,
                        "trace_seed": trace_seed,
                        "total_cost": total_cost,
                        "planning_ms_per_decision": planning_ms / horizon,
                    }
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(results / "planner_depth_raw.csv", index=False)
    summary = raw.groupby("planner_horizon").agg(
        total_cost_mean=("total_cost", "mean"),
        total_cost_sd=("total_cost", "std"),
        planning_ms_mean=("planning_ms_per_decision", "mean"),
        planning_ms_sd=("planning_ms_per_decision", "std"),
        n_traces=("trace_seed", "nunique"),
    ).reset_index()
    summary.to_csv(results / "planner_depth_summary.csv", index=False)
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default="800-819")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--test-traces", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stage", choices=("train", "evaluate", "planning", "all"), default="all"
    )
    args = parser.parse_args()
    start, end = (int(value) for value in args.seeds.split("-", 1))
    seeds = tuple(range(start, end + 1))
    args.results.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents: dict[tuple[str, int], object] = {}
    if args.stage in {"train", "evaluate", "all"}:
        agents = train_extended(
            args.results, seeds, args.episodes, args.horizon, device, args.resume
        )
        write_model_manifest(args.results)
    if args.stage in {"evaluate", "all"}:
        evaluate_ablations(
            args.results, agents, seeds, args.horizon, args.test_traces
        )
        primary_path = args.results / "seven_regimes_raw.csv"
        if not primary_path.exists():
            primary_path = DEFAULT_RESULTS / "seven_regimes_raw.csv"
        primary = pd.read_csv(primary_path)
        unpaired_seed_bootstrap(primary).to_csv(
            args.results / "primary_unpaired_seed_bootstrap.csv", index=False
        )
    if args.stage in {"planning", "all"}:
        evaluate_planner_depth(
            args.results, seeds, args.horizon, args.test_traces
        )

    metadata = {
        "variants": list(ALL_LEARNED),
        "new_variants": list(EXTENDED_VARIANTS),
        "model_seeds": list(seeds),
        "episodes": args.episodes,
        "horizon": args.horizon,
        "test_traces_per_seed_namespace": args.test_traces,
        "common_random_numbers": {
            "network_initialization": True,
            "training_environment_episode_seeds": True,
            "epsilon_exploration_stream": True,
            "replay_index_stream": True,
            "held_out_trace_seeds": True,
        },
        "planner_horizons": list(PLANNER_HORIZONS),
        "auxiliary_lambda": AUXILIARY_LAMBDA,
    }
    (args.results / "extended_revision_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.stage in {"evaluate", "all"}:
        raw = pd.read_csv(args.results / "extended_ablation_raw.csv")
        expected = len(FULL_COUPLING) * len(ALL_LEARNED) * len(seeds) * args.test_traces
        if len(raw) != expected or not np.isfinite(
            raw.select_dtypes(include=[np.number])
        ).all().all():
            raise AssertionError("extended ablation validation failed")
        checkpoints = []
        for seed in seeds:
            for variant in ALL_LEARNED:
                path = args.results / "models" / {
                    "standard_dqn": f"standard_dqn__standard__seed_{seed}.pth",
                    "full_action_q_dqn": f"full_action_q_dqn__lambda_0p3_uncentered__seed_{seed}.pth",
                    "immediate_advantage_dqn": f"immediate_advantage_dqn__lambda_0p3_immediate__seed_{seed}.pth",
                    "centered_full_action_dqn": f"centered_full_action_dqn__lambda_0p3__seed_{seed}.pth",
                    "double_dqn": f"double_dqn__double__seed_{seed}.pth",
                    "double_centered_full_action_dqn": f"double_centered_full_action_dqn__double_lambda_0p3__seed_{seed}.pth",
                }[variant]
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if int(payload["training_steps"]) != args.episodes * args.horizon:
                    raise AssertionError(f"training budget mismatch for {path.name}")
                checkpoints.append(
                    {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                )
        validation = {
            "status": "passed",
            "expected_rows": expected,
            "observed_rows": len(raw),
            "all_numeric_finite": True,
            "checkpoint_count": len(checkpoints),
            "checkpoint_sha256": checkpoints,
        }
        (args.results / "extended_revision_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
