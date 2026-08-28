"""Reviewer-driven internally consistent simulation and convergence study.

This module writes to ``results/reviewer_revision_state_complete`` and never overwrites the
locked confirmation artifacts used by the preceding manuscript version.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from .agent import DQNAgent
from .environment import EnvConfig, SatelliteSchedulingEnv
from .experiment import (
    AUXILIARY_LAMBDA,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EPISODES,
    FULL_COUPLING,
    HORIZON,
    SCENARIOS,
    curve_path,
    load_checkpoint,
    save_checkpoint,
    threshold_action,
    train_model,
)
from .planner import PlannerConfig, RecedingHorizonPlanner


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "reviewer_revision_state_complete"
TRACE_BASE = 3_300_000
CONFIRMATION_SEEDS = tuple(range(800, 820))
LEARNED = ("contextual_bandit", "standard_dqn", "centered_full_action_dqn")
POLICIES = (
    "immediate_argmin",
    "mpc_h4",
    "contextual_bandit",
    "threshold",
    "standard_dqn",
    "centered_full_action_dqn",
    "fixed_onboard",
    "fixed_ground",
)
MAIN_POLICIES = POLICIES[:7]
ADDITIVE_METRICS = (
    "total_cost",
    "immediate_cost",
    "penalty",
    "energy_use",
    "latency_ms",
    "transmitted_mbit",
    "thermal_violation",
    "energy_violation",
    "queue_violation",
    "contact_violation",
    "planning_time_ms",
)
METRICS = ADDITIVE_METRICS + (
    "maximum_heat",
    "minimum_energy",
    "threshold_free_episode",
    "onboard_fraction",
    "ground_fraction",
    "hybrid_fraction",
)
SENSITIVITY_PROFILES = {
    "reference": {},
    "compute_minus20": {"compute_scale": 0.8},
    "compute_plus20": {"compute_scale": 1.2},
    "data_minus20": {"data_scale": 0.8},
    "data_plus20": {"data_scale": 1.2},
    "energy_plus20": {"energy_scale": 1.2},
    "heat_plus20": {"heat_scale": 1.2},
    "link_minus20": {"link_scale": 0.8},
}
PREVIEW_MISMATCH_SCALES = (0.85, 1.15)


def revision_env(horizon: int, scenario: str = "nominal", coupling: float = 1.0,
                 **overrides: float) -> EnvConfig:
    return EnvConfig(
        horizon=horizon,
        scenario=scenario,
        coupling=coupling,
        generator="physics_correlated",
        **overrides,
    )


def revision_model_path(results: Path, variant: str, seed: int) -> Path:
    tag = {
        "contextual_bandit": "gamma_0",
        "standard_dqn": "standard",
        "centered_full_action_dqn": "lambda_0p3",
        "full_action_q_dqn": "lambda_0p3_uncentered",
        "immediate_advantage_dqn": "lambda_0p3_immediate",
        "double_dqn": "double",
        "double_centered_full_action_dqn": "double_lambda_0p3",
    }[variant]
    return results / "models" / f"{variant}__{tag}__seed_{seed}.pth"


def mismatch_model_path(results: Path, scale: float, seed: int) -> Path:
    tag = str(scale).replace(".", "p")
    return results / "mismatch_models" / f"centered_full_action_dqn__preview_scale_{tag}__seed_{seed}.pth"


def train_or_load_revision(
    results: Path,
    variant: str,
    seed: int,
    episodes: int,
    horizon: int,
    device: torch.device,
    resume: bool,
) -> DQNAgent:
    path = revision_model_path(results, variant, seed)
    if resume and path.exists():
        return load_checkpoint(
            path, variant, seed, device, episodes, horizon, AUXILIARY_LAMBDA
        )
    agent, curve = train_model(
        variant,
        seed,
        episodes,
        horizon,
        device,
        AUXILIARY_LAMBDA,
        environment_config=revision_env(horizon),
    )
    save_checkpoint(path, agent, seed, episodes, horizon)
    pd.DataFrame(curve).to_csv(curve_path(path), index=False)
    return agent


def collect_curves(results: Path) -> pd.DataFrame:
    paths = sorted((results / "models").glob("*__curve.csv"))
    if not paths:
        raise FileNotFoundError("no revision training curves found")
    curves = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    curves.to_csv(results / "training_curves.csv", index=False)
    return curves


def write_model_manifest(results: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted((results / "models").glob("*.pth")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "file": path.name,
                "sha256": digest,
                "variant": payload["variant"],
                "model_seed": payload["model_seed"],
                "episodes": payload["episodes"],
                "horizon": payload["horizon"],
                "training_steps": payload["training_steps"],
                "state_dim": payload["state_dim"],
                "bellman_target": payload["bellman_target"],
                "replay": payload["replay"],
                "training_wall_time_s": (
                    payload.get("training_accounting") or {}
                ).get("wall_time_s"),
                "optimizer_updates": (
                    payload.get("training_accounting") or {}
                ).get("optimizer_updates"),
                "preview_calls": (
                    payload.get("training_accounting") or {}
                ).get("preview_calls"),
                "preview_action_evaluations": (
                    payload.get("training_accounting") or {}
                ).get("preview_action_evaluations"),
            }
        )
    manifest = pd.DataFrame(rows)
    if len(manifest) == 0:
        raise FileNotFoundError("no revision checkpoints found")
    if not (manifest.training_steps == manifest.episodes * manifest.horizon).all():
        raise AssertionError("checkpoint interaction budget mismatch")
    manifest.to_csv(results / "model_manifest.csv", index=False)
    return manifest


def evaluate_policy(
    policy: str,
    agent: DQNAgent | None,
    model_seed: int,
    trace_seeds: list[int],
    config: EnvConfig,
    scenario_label: str,
) -> list[dict]:
    planner = RecedingHorizonPlanner(PlannerConfig(horizon=4, gamma=0.97))
    rows: list[dict] = []
    for trace_seed in trace_seeds:
        env = SatelliteSchedulingEnv(config)
        state = env.reset(seed=trace_seed)
        totals = {key: 0.0 for key in ADDITIVE_METRICS}
        maximum_heat = float("-inf")
        minimum_energy = float("inf")
        actions = np.zeros(3, dtype=int)
        done = False
        while not done:
            costs = env.immediate_costs()
            planning_ms = 0.0
            if policy == "immediate_argmin":
                action = int(np.argmin(costs))
            elif policy == "mpc_h4":
                started = time.perf_counter()
                action = planner.act(env)
                planning_ms = 1000.0 * (time.perf_counter() - started)
            elif policy == "threshold":
                action = threshold_action(state, costs)
            elif policy == "fixed_onboard":
                action = 0
            elif policy == "fixed_ground":
                action = 1
            else:
                if agent is None:
                    raise ValueError(f"missing trained agent for {policy}")
                action = agent.act(state, explore=False)
            state, _, done, info = env.step(action)
            actions[action] += 1
            for key in ADDITIVE_METRICS:
                if key == "planning_time_ms":
                    continue
                totals[key] += info[key]
            totals["planning_time_ms"] += planning_ms
            maximum_heat = max(maximum_heat, float(info["heat"]))
            minimum_energy = min(minimum_energy, float(info["energy"]))
        threshold_exposures = sum(
            totals[key]
            for key in (
                "thermal_violation",
                "energy_violation",
                "queue_violation",
                "contact_violation",
            )
        )
        rows.append(
            {
                "policy": policy,
                "model_seed": model_seed,
                "trace_seed": trace_seed,
                "scenario": scenario_label,
                "environment_scenario": config.scenario,
                "coupling": config.coupling,
                "generator": config.generator,
                "compute_scale": config.compute_scale,
                "data_scale": config.data_scale,
                "energy_scale": config.energy_scale,
                "heat_scale": config.heat_scale,
                "link_scale": config.link_scale,
                **totals,
                "maximum_heat": maximum_heat,
                "minimum_energy": minimum_energy,
                "threshold_free_episode": float(threshold_exposures == 0),
                "onboard_fraction": actions[0] / config.horizon,
                "ground_fraction": actions[1] / config.horizon,
                "hybrid_fraction": actions[2] / config.horizon,
            }
        )
    return rows


def seed_level(raw: pd.DataFrame, group: str = "scenario") -> pd.DataFrame:
    return raw.groupby([group, "policy", "model_seed"], as_index=False)[
        list(METRICS)
    ].mean()


def summarize(raw: pd.DataFrame, group: str = "scenario") -> pd.DataFrame:
    by_seed = seed_level(raw, group)
    grouped = by_seed.groupby([group, "policy"])[list(METRICS)]
    result = pd.concat(
        [
            grouped.mean().add_suffix("_mean"),
            grouped.std(ddof=1).add_suffix("_sd"),
            grouped.size().rename("n_model_seeds"),
        ],
        axis=1,
    ).reset_index()
    return result


def _exact_two_sided_sign_p(difference: np.ndarray) -> float:
    nonzero = difference[np.abs(difference) > np.finfo(float).eps]
    n = len(nonzero)
    if n == 0:
        return 1.0
    negative = int((nonzero < 0).sum())
    positive = n - negative
    tail = min(negative, positive)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, probability))


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_bootstrap(
    raw: pd.DataFrame,
    comparisons: tuple[tuple[str, str], ...],
    group: str = "scenario",
    inference_family: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    by_seed = seed_level(raw, group)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    for label in dict.fromkeys(by_seed[group].tolist()):
        pivot = by_seed[by_seed[group] == label].pivot(
            index="model_seed", columns="policy", values="total_cost"
        )
        for treatment, reference in comparisons:
            difference = (pivot[treatment] - pivot[reference]).dropna().to_numpy()
            boot = rng.choice(
                difference,
                size=(BOOTSTRAP_RESAMPLES, len(difference)),
                replace=True,
            ).mean(axis=1)
            reference_mean = float(pivot[reference].mean())
            leave_one_out = np.array(
                [np.delete(difference, index).mean() for index in range(len(difference))]
            )
            rows.append(
                {
                    group: label,
                    "comparison": f"{treatment} - {reference}",
                    "n_pairs": len(difference),
                    "mean_difference": float(difference.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "relative_improvement_percent": float(
                        -100.0 * difference.mean() / reference_mean
                    ),
                    "exact_sign_p": _exact_two_sided_sign_p(difference),
                    "negative_seed_pairs": int((difference < 0).sum()),
                    "positive_seed_pairs": int((difference > 0).sum()),
                    "loo_mean_min": float(leave_one_out.min()),
                    "loo_mean_max": float(leave_one_out.max()),
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = np.nan
    family = set(inference_family or ())
    for comparison in result.comparison.unique():
        mask = result.comparison.eq(comparison)
        if family:
            mask &= result[group].isin(family)
        if mask.any():
            result.loc[mask, "holm_adjusted_p"] = _holm_adjust(
                result.loc[mask, "exact_sign_p"].to_numpy()
            )
    result["in_prespecified_family"] = result[group].isin(family) if family else True
    result["mean_better"] = result.mean_difference < 0
    result["supported"] = (
        result.mean_better
        & (result.ci95_high < 0)
        & (result.holm_adjusted_p < 0.05)
    )
    return result


def paired_seed_differences(
    raw: pd.DataFrame,
    comparisons: tuple[tuple[str, str], ...],
    group: str = "scenario",
) -> pd.DataFrame:
    by_seed = seed_level(raw, group)
    rows: list[dict] = []
    for label in dict.fromkeys(by_seed[group].tolist()):
        pivot = by_seed[by_seed[group] == label].pivot(
            index="model_seed", columns="policy", values="total_cost"
        )
        for treatment, reference in comparisons:
            for model_seed, value in (pivot[treatment] - pivot[reference]).dropna().items():
                rows.append(
                    {
                        group: label,
                        "comparison": f"{treatment} - {reference}",
                        "model_seed": int(model_seed),
                        "difference": float(value),
                    }
                )
    return pd.DataFrame(rows)


def train_revision_models(
    results: Path,
    seeds: tuple[int, ...],
    episodes: int,
    horizon: int,
    device: torch.device,
    resume: bool,
) -> dict[tuple[str, int], DQNAgent]:
    agents: dict[tuple[str, int], DQNAgent] = {}
    for seed in seeds:
        for variant in LEARNED:
            print(f"training/loading {variant} seed={seed}", flush=True)
            agents[(variant, seed)] = train_or_load_revision(
                results, variant, seed, episodes, horizon, device, resume
            )
    collect_curves(results)
    write_model_manifest(results)
    return agents


def load_revision_models(
    results: Path,
    seeds: tuple[int, ...],
    episodes: int,
    horizon: int,
    device: torch.device,
) -> dict[tuple[str, int], DQNAgent]:
    return {
        (variant, seed): load_checkpoint(
            revision_model_path(results, variant, seed),
            variant,
            seed,
            device,
            episodes,
            horizon,
            AUXILIARY_LAMBDA,
        )
        for seed in seeds
        for variant in LEARNED
    }


def train_or_load_mismatch_model(
    results: Path,
    scale: float,
    seed: int,
    episodes: int,
    horizon: int,
    device: torch.device,
    resume: bool,
) -> DQNAgent:
    path = mismatch_model_path(results, scale, seed)
    if resume and path.exists():
        return load_checkpoint(
            path, "centered_full_action_dqn", seed, device, episodes, horizon, AUXILIARY_LAMBDA
        )
    agent, curve = train_model(
        "centered_full_action_dqn",
        seed,
        episodes,
        horizon,
        device,
        AUXILIARY_LAMBDA,
        environment_config=revision_env(horizon),
        preview_transition_scale=scale,
    )
    save_checkpoint(path, agent, seed, episodes, horizon)
    curve_frame = pd.DataFrame(curve)
    curve_frame["preview_transition_scale"] = scale
    path.parent.mkdir(parents=True, exist_ok=True)
    curve_frame.to_csv(curve_path(path), index=False)
    return agent


def write_mismatch_model_manifest(results: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((results / "mismatch_models").glob("*.pth")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_seed": payload["model_seed"],
                "training_steps": payload["training_steps"],
                "preview_transition_scale": 0.85 if "0p85" in path.name else 1.15,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(results / "preview_mismatch_model_manifest.csv", index=False)
    return manifest


def run_model_mismatch(
    results: Path,
    agents: dict[tuple[str, int], DQNAgent],
    seeds: tuple[int, ...],
    episodes: int,
    horizon: int,
    test_traces: int,
    device: torch.device,
    resume: bool,
) -> pd.DataFrame:
    mismatch_agents: dict[tuple[float, int], DQNAgent] = {}
    for seed in seeds:
        for scale in PREVIEW_MISMATCH_SCALES:
            print(f"training/loading centered full-action preview scale={scale} seed={seed}", flush=True)
            mismatch_agents[(scale, seed)] = train_or_load_mismatch_model(
                results, scale, seed, episodes, horizon, device, resume
            )
    write_mismatch_model_manifest(results)
    rows: list[dict] = []
    for seed_index, seed in enumerate(seeds):
        traces = [TRACE_BASE + seed_index * 1000 + i for i in range(test_traces)]
        for label, scenario, coupling in FULL_COUPLING:
            config = revision_env(horizon, scenario=scenario, coupling=coupling)
            policies = (
                ("standard_dqn", agents[("standard_dqn", seed)]),
                ("centered_full_action_dqn_exact", agents[("centered_full_action_dqn", seed)]),
                ("centered_full_action_dqn_preview_0p85", mismatch_agents[(0.85, seed)]),
                ("centered_full_action_dqn_preview_1p15", mismatch_agents[(1.15, seed)]),
            )
            for policy, agent in policies:
                rows.extend(evaluate_policy(policy, agent, seed, traces, config, label))
    raw = pd.DataFrame(rows)
    raw.to_csv(results / "preview_mismatch_raw.csv", index=False)
    seed_level(raw).to_csv(results / "preview_mismatch_seed_level.csv", index=False)
    summarize(raw).to_csv(results / "preview_mismatch_summary.csv", index=False)
    comparisons = (
        ("centered_full_action_dqn_exact", "standard_dqn"),
        ("centered_full_action_dqn_preview_0p85", "standard_dqn"),
        ("centered_full_action_dqn_preview_1p15", "standard_dqn"),
    )
    paired_bootstrap(
        raw,
        comparisons,
        inference_family=tuple(item[0] for item in FULL_COUPLING),
    ).to_csv(results / "preview_mismatch_inference.csv", index=False)
    paired_seed_differences(raw, comparisons).to_csv(
        results / "preview_mismatch_seed_differences.csv", index=False
    )
    return raw


def run_seven_regimes(
    results: Path,
    agents: dict[tuple[str, int], DQNAgent],
    seeds: tuple[int, ...],
    horizon: int,
    test_traces: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed_index, seed in enumerate(seeds):
        traces = [TRACE_BASE + seed_index * 1000 + i for i in range(test_traces)]
        for label, scenario, coupling in SCENARIOS:
            config = revision_env(horizon, scenario=scenario, coupling=coupling)
            for policy in POLICIES:
                rows.extend(
                    evaluate_policy(
                        policy, agents.get((policy, seed)), seed, traces, config, label
                    )
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(results / "seven_regimes_raw.csv", index=False)
    seed_level(raw).to_csv(results / "seven_regimes_seed_level.csv", index=False)
    summarize(raw).to_csv(results / "seven_regimes_summary.csv", index=False)
    comparisons = (
        ("centered_full_action_dqn", "standard_dqn"),
        ("centered_full_action_dqn", "mpc_h4"),
        ("mpc_h4", "immediate_argmin"),
    )
    paired_bootstrap(
        raw,
        comparisons,
        inference_family=tuple(item[0] for item in FULL_COUPLING),
    ).to_csv(results / "seven_regimes_paired_bootstrap.csv", index=False)
    paired_seed_differences(raw, comparisons).to_csv(
        results / "seven_regimes_paired_seed_differences.csv", index=False
    )
    return raw


def run_sensitivity(
    results: Path,
    agents: dict[tuple[str, int], DQNAgent],
    seeds: tuple[int, ...],
    horizon: int,
    test_traces: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    policies = ("immediate_argmin", "mpc_h4", "standard_dqn", "centered_full_action_dqn")
    for seed_index, seed in enumerate(seeds):
        traces = [TRACE_BASE + seed_index * 1000 + i for i in range(test_traces)]
        for profile, overrides in SENSITIVITY_PROFILES.items():
            config = revision_env(horizon, **overrides)
            for policy in policies:
                profile_rows = evaluate_policy(
                    policy, agents.get((policy, seed)), seed, traces, config, profile
                )
                for row in profile_rows:
                    row["profile"] = profile
                rows.extend(profile_rows)
    raw = pd.DataFrame(rows)
    raw.to_csv(results / "sensitivity_raw.csv", index=False)
    seed_level(raw, "profile").to_csv(
        results / "sensitivity_seed_level.csv", index=False
    )
    summarize(raw, "profile").to_csv(results / "sensitivity_summary.csv", index=False)
    comparisons = (("centered_full_action_dqn", "standard_dqn"), ("centered_full_action_dqn", "mpc_h4"))
    paired_bootstrap(
        raw,
        comparisons,
        "profile",
        inference_family=tuple(SENSITIVITY_PROFILES),
    ).to_csv(results / "sensitivity_paired_bootstrap.csv", index=False)
    paired_seed_differences(raw, comparisons, "profile").to_csv(
        results / "sensitivity_paired_seed_differences.csv", index=False
    )
    return raw


def audit_parameter_generator(results: Path) -> dict:
    env = SatelliteSchedulingEnv(revision_env(20_000))
    env.reset(seed=9_117_031)
    primitive_names = (
        "raw_mbit",
        "workload_gflop",
        "result_ratio",
        "feature_ratio",
        "preprocess_fraction",
        "compute_power_w",
        "onboard_rate_gflops",
        "ground_rate_gflops",
        "reference_link_mbps",
    )
    primitives = pd.DataFrame(env.task_primitives, columns=primitive_names)
    descriptors = pd.DataFrame(
        {
            "result_mbit": env.tasks[:, 5],
            "feature_mbit": env.tasks[:, 14],
            "onboard_heat_j": env.tasks[:, 0],
            "ground_tx_heat_j": env.tasks[:, 6],
            "onboard_latency_ms": env.tasks[:, 2],
            "ground_compute_ms": env.tasks[:, 7],
        }
    )
    sample = pd.concat([primitives, descriptors], axis=1)
    sample.to_csv(results / "parameter_audit_sample.csv", index=False)
    summary = sample.agg(["min", "median", "max"]).T.reset_index()
    summary.columns = ["parameter", "minimum", "median", "maximum"]
    summary.to_csv(results / "parameter_ranges_empirical.csv", index=False)
    sample.corr(numeric_only=True).to_csv(results / "parameter_correlations.csv")
    validity = {
        "n_tasks": int(len(sample)),
        "finite": bool(np.isfinite(sample.to_numpy()).all()),
        "result_le_feature": bool((sample.result_mbit <= sample.feature_mbit).all()),
        "feature_le_raw": bool((sample.feature_mbit <= sample.raw_mbit).all()),
        "positive_primitives": bool((sample[list(primitive_names)] > 0).all().all()),
        "corr_raw_workload": float(sample.raw_mbit.corr(sample.workload_gflop)),
        "corr_workload_heat": float(
            sample.workload_gflop.corr(sample.onboard_heat_j)
        ),
        "corr_raw_ground_tx_heat": float(
            sample.raw_mbit.corr(sample.ground_tx_heat_j)
        ),
        "generator": "physics_correlated",
        "seed": 9_117_031,
    }
    if not all(
        validity[key]
        for key in ("finite", "result_le_feature", "feature_le_raw", "positive_primitives")
    ):
        raise AssertionError(f"physical validity gate failed: {validity}")
    (results / "parameter_validity.json").write_text(
        json.dumps(validity, indent=2), encoding="utf-8"
    )
    return validity


def write_metadata(
    results: Path,
    seeds: tuple[int, ...],
    episodes: int,
    horizon: int,
    test_traces: int,
) -> None:
    metadata = {
        "training_environment": asdict(revision_env(horizon)),
        "episodes": episodes,
        "horizon": horizon,
        "real_steps_per_model": episodes * horizon,
        "state_definition": {
            "dimension": SatelliteSchedulingEnv.state_dim,
            "task_descriptors": SatelliteSchedulingEnv.task_dim,
            "context": ["normalized_time", "sin_link_phase", "cos_link_phase"],
            "resource_coordinates": SatelliteSchedulingEnv.resource_dim,
        },
        "model_seeds": list(seeds),
        "test_traces_per_seed": test_traces,
        "planner": asdict(PlannerConfig(horizon=4, gamma=0.97)),
        "planner_forecast": "perfect next-four-task and link-state forecast",
        "sensitivity_profiles": SENSITIVITY_PROFILES,
        "preview_mismatch_transition_scales": list(PREVIEW_MISMATCH_SCALES),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "independent_unit": "independently trained model seed",
        "primary_inference_family": [item[0] for item in FULL_COUPLING],
        "multiplicity": "Holm adjustment of exact two-sided paired sign tests",
        "backbone": "standard DQN; no DDQN, dueling, PER, n-step, noisy, or distributional components",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    (results / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=(
            "train", "evaluate", "sensitivity", "model-mismatch", "audit", "validate", "all"
        )
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--test-traces", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--seeds",
        type=str,
        help="Optional comma-separated seeds or inclusive range, for example 800-819.",
    )
    return parser.parse_args()


def parse_seeds(value: str | None) -> tuple[int, ...]:
    if value is None:
        return CONFIRMATION_SEEDS
    seeds: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if "-" in token:
            left, right = token.split("-", 1)
            seeds.extend(range(int(left), int(right) + 1))
        else:
            seeds.append(int(token))
    unique = tuple(dict.fromkeys(seeds))
    if not unique:
        raise ValueError("at least one model seed is required")
    return unique


def write_revision_validation(
    results: Path,
    seeds: tuple[int, ...],
    test_traces: int,
) -> None:
    raw = pd.read_csv(results / "seven_regimes_raw.csv")
    sensitivity = pd.read_csv(results / "sensitivity_raw.csv")
    manifest = pd.read_csv(results / "model_manifest.csv")
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    expected_steps = int(metadata["real_steps_per_model"])
    expected_main = len(SCENARIOS) * len(POLICIES) * len(seeds) * test_traces
    expected_sensitivity = len(SENSITIVITY_PROFILES) * 4 * len(seeds) * test_traces
    primary_manifest = manifest[manifest.variant.isin(LEARNED)].copy()
    cells = raw.groupby(["scenario", "policy", "model_seed"]).size()
    if len(raw) != expected_main or len(sensitivity) != expected_sensitivity:
        raise AssertionError("reviewer-revision row count mismatch")
    if len(cells) != len(SCENARIOS) * len(POLICIES) * len(seeds):
        raise AssertionError("reviewer-revision cell count mismatch")
    if not (cells == test_traces).all():
        raise AssertionError("unequal test-trace count in reviewer revision")
    if len(primary_manifest) != len(LEARNED) * len(seeds):
        raise AssertionError("required primary checkpoint manifest count mismatch")
    required_cells = primary_manifest.groupby(["variant", "model_seed"]).size()
    if len(required_cells) != len(LEARNED) * len(seeds) or not (
        required_cells == 1
    ).all():
        raise AssertionError("required primary checkpoint cells mismatch")
    if set(primary_manifest.model_seed.astype(int)) != set(seeds):
        raise AssertionError("required primary checkpoint seed mismatch")
    if not (manifest.training_steps.astype(int) == expected_steps).all():
        raise AssertionError("checkpoint training-budget mismatch")
    main_finite = bool(np.isfinite(raw.select_dtypes(include=[np.number])).all().all())
    sensitivity_finite = bool(
        np.isfinite(sensitivity.select_dtypes(include=[np.number])).all().all()
    )
    if not main_finite or not sensitivity_finite:
        raise AssertionError("non-finite reviewer-revision evaluation value")
    primary_nominal = raw[
        raw.scenario.eq("nominal")
        & raw.policy.isin(("standard_dqn", "centered_full_action_dqn"))
    ].sort_values(["policy", "model_seed", "trace_seed"])
    sensitivity_reference = sensitivity[
        sensitivity.profile.eq("reference")
        & sensitivity.policy.isin(("standard_dqn", "centered_full_action_dqn"))
    ].sort_values(["policy", "model_seed", "trace_seed"])
    if not np.array_equal(
        primary_nominal.total_cost.to_numpy(),
        sensitivity_reference.total_cost.to_numpy(),
    ):
        raise AssertionError("reference sensitivity does not reuse primary nominal traces")
    artifacts = {}
    names = [
        "seven_regimes_raw.csv",
        "seven_regimes_seed_level.csv",
        "seven_regimes_summary.csv",
        "seven_regimes_paired_bootstrap.csv",
        "seven_regimes_paired_seed_differences.csv",
        "sensitivity_raw.csv",
        "sensitivity_seed_level.csv",
        "sensitivity_summary.csv",
        "sensitivity_paired_bootstrap.csv",
        "sensitivity_paired_seed_differences.csv",
        "model_manifest.csv",
        "metadata.json",
    ]
    mismatch_path = results / "preview_mismatch_raw.csv"
    mismatch_rows = None
    mismatch_checkpoints = None
    mismatch_finite = None
    if mismatch_path.exists():
        mismatch = pd.read_csv(mismatch_path)
        expected_mismatch = len(FULL_COUPLING) * 4 * len(seeds) * test_traces
        if len(mismatch) != expected_mismatch:
            raise AssertionError("preview-mismatch row count mismatch")
        mismatch_manifest = pd.read_csv(results / "preview_mismatch_model_manifest.csv")
        if len(mismatch_manifest) != len(PREVIEW_MISMATCH_SCALES) * len(seeds):
            raise AssertionError("preview-mismatch checkpoint count mismatch")
        if set(mismatch_manifest.model_seed.astype(int)) != set(seeds):
            raise AssertionError("preview-mismatch checkpoint seed mismatch")
        if not (
            mismatch_manifest.training_steps.astype(int) == expected_steps
        ).all():
            raise AssertionError("preview-mismatch training-budget mismatch")
        mismatch_finite = bool(
            np.isfinite(mismatch.select_dtypes(include=[np.number])).all().all()
        )
        if not mismatch_finite:
            raise AssertionError("non-finite preview-mismatch evaluation value")
        mismatch_exact = mismatch[
            mismatch.scenario.eq("nominal")
            & mismatch.policy.isin(("standard_dqn", "centered_full_action_dqn_exact"))
        ].copy()
        mismatch_exact["policy"] = mismatch_exact.policy.replace(
            {"centered_full_action_dqn_exact": "centered_full_action_dqn"}
        )
        mismatch_exact = mismatch_exact.sort_values(
            ["policy", "model_seed", "trace_seed"]
        )
        if not np.array_equal(
            primary_nominal.total_cost.to_numpy(),
            mismatch_exact.total_cost.to_numpy(),
        ):
            raise AssertionError("exact-preview table does not reuse primary nominal traces")
        mismatch_rows = len(mismatch)
        mismatch_checkpoints = len(mismatch_manifest)
        names.extend(
            [
                "preview_mismatch_raw.csv",
                "preview_mismatch_seed_level.csv",
                "preview_mismatch_summary.csv",
                "preview_mismatch_inference.csv",
                "preview_mismatch_seed_differences.csv",
                "preview_mismatch_model_manifest.csv",
            ]
        )
    for name in (
        "dqn_vs_mpc_inference.csv",
        "dqn_vs_mpc_family_inference.csv",
        "dqn_vs_mpc_seed_differences.csv",
    ):
        if (results / name).exists():
            names.append(name)
    for name in names:
        path = results / name
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_files = (
        ROOT / "agent.py",
        ROOT / "environment.py",
        ROOT / "experiment.py",
        ROOT / "extended_revision.py",
        ROOT / "transition_audit.py",
        ROOT / "planner.py",
        ROOT / "reviewer_experiments.py",
        ROOT / "reviewer_reporting.py",
        ROOT / "dqn_mpc_family_inference.py",
        ROOT / "extended_reporting.py",
        ROOT / "plot_reviewer_results_matlab.m",
        ROOT / "plot_dqn_family_summary.py",
        ROOT / "requirements.txt",
        ROOT.parent / "dqn_family_satellite_ground_manuscript.pdf",

    )
    source_hashes = {
        str(path.relative_to(ROOT.parent)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    payload = {
        "status": "passed",
        "model_seeds": list(seeds),
        "n_independent_model_seeds": len(seeds),
        "test_traces_per_seed": test_traces,
        "seven_regimes_rows": len(raw),
        "seven_regimes_policy_seed_cells": len(cells),
        "sensitivity_rows": len(sensitivity),
        "required_primary_checkpoints": len(primary_manifest),
        "all_manifest_checkpoints": len(manifest),
        "expected_real_steps_per_checkpoint": expected_steps,
        "seven_regimes_all_numeric_finite": main_finite,
        "sensitivity_all_numeric_finite": sensitivity_finite,
        "preview_mismatch_rows": mismatch_rows,
        "preview_mismatch_checkpoints": mismatch_checkpoints,
        "preview_mismatch_all_numeric_finite": mismatch_finite,
        "nominal_reference_rows_identical_across_analyses": True,
        "artifacts_sha256": artifacts,
        "source_sha256": source_hashes,
    }
    (results / "validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if args.smoke:
        args.episodes, args.horizon, args.test_traces = 12, 16, 3
        seeds = (CONFIRMATION_SEEDS[0],)
        if args.results == DEFAULT_RESULTS:
            args.results = ROOT / "results" / "reviewer_revision_state_complete_smoke"
    args.results.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"stage={args.stage} device={device} results={args.results}", flush=True)
    audit_parameter_generator(args.results)
    write_metadata(args.results, seeds, args.episodes, args.horizon, args.test_traces)
    if args.stage == "validate":
        write_revision_validation(args.results, seeds, args.test_traces)
        return
    agents: dict[tuple[str, int], DQNAgent] | None = None
    if args.stage in {"train", "all"}:
        agents = train_revision_models(
            args.results,
            seeds,
            args.episodes,
            args.horizon,
            device,
            args.resume,
        )
        if args.stage == "train":
            return
    if args.stage == "audit":
        return
    if agents is None:
        agents = load_revision_models(
            args.results, seeds, args.episodes, args.horizon, device
        )
    if args.stage in {"evaluate", "all"}:
        raw = run_seven_regimes(
            args.results, agents, seeds, args.horizon, args.test_traces
        )
        expected = len(SCENARIOS) * len(POLICIES) * len(seeds) * args.test_traces
        if len(raw) != expected:
            raise AssertionError(f"expected {expected} evaluation rows, found {len(raw)}")
        if args.stage == "evaluate":
            return
    if args.stage in {"sensitivity", "all"}:
        raw = run_sensitivity(
            args.results, agents, seeds, args.horizon, args.test_traces
        )
        expected = len(SENSITIVITY_PROFILES) * 4 * len(seeds) * args.test_traces
        if len(raw) != expected:
            raise AssertionError(f"expected {expected} sensitivity rows, found {len(raw)}")
    if args.stage in {"model-mismatch", "all"}:
        raw = run_model_mismatch(
            args.results,
            agents,
            seeds,
            args.episodes,
            args.horizon,
            args.test_traces,
            device,
            args.resume,
        )
        expected = len(FULL_COUPLING) * 4 * len(seeds) * args.test_traces
        if len(raw) != expected:
            raise AssertionError(f"expected {expected} mismatch rows, found {len(raw)}")
    if args.stage == "all":
        write_revision_validation(args.results, seeds, args.test_traces)


if __name__ == "__main__":
    main()
