"""Unified experiment CLI for the locked DQN-family implementation study.

Examples
--------
python -m dqn_family_satellite_ground.experiment seven-scenarios --resume
python -m dqn_family_satellite_ground.experiment all --resume
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .agent import AgentConfig, DQNAgent
from .environment import EnvConfig, SatelliteSchedulingEnv


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
EPISODES = 600
HORIZON = 64
AUXILIARY_LAMBDA = 0.3
MODEL_SEEDS = tuple(range(800, 805))
CALIBRATION_SEEDS = tuple(range(700, 704))
TRACE_BASE = 2_300_000
BOOTSTRAP_SEED = 20_260_829
BOOTSTRAP_RESAMPLES = 10_000

# Public labels deliberately distinguish coupling controls from operating stressors.
SCENARIOS = (
    ("static", "nominal", 0.0),
    ("reduced_coupling", "nominal", 0.5),
    ("nominal", "nominal", 1.0),
    ("burst", "burst", 1.0),
    ("link_limited", "link_limited", 1.0),
    ("energy_limited", "energy_limited", 1.0),
    ("thermal_stress", "thermal_stress", 1.0),
)
FULL_COUPLING = tuple(item for item in SCENARIOS if item[2] == 1.0)
LEARNED = ("contextual_bandit", "standard_dqn", "centered_full_action_dqn")
HEURISTICS = ("immediate_argmin", "threshold", "fixed_onboard", "fixed_ground")
ALL_POLICIES = HEURISTICS + LEARNED
MAIN_POLICIES = (
    "immediate_argmin",
    "contextual_bandit",
    "threshold",
    "standard_dqn",
    "centered_full_action_dqn",
)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_for(variant: str, episodes: int = EPISODES, horizon: int = HORIZON,
               auxiliary_lambda: float = AUXILIARY_LAMBDA) -> AgentConfig:
    gamma = 0.0 if variant == "contextual_bandit" else 0.97
    return AgentConfig(
        gamma=gamma,
        epsilon_decay_steps=max(1_000, int(episodes * horizon * 0.35)),
        auxiliary_lambda=auxiliary_lambda,
    )


def model_path(results: Path, stage: str, variant: str, seed: int) -> Path:
    configuration = {
        "standard_dqn": "standard",
        "centered_full_action_dqn": "lambda_0p3",
        "full_action_q_dqn": "lambda_0p3_uncentered",
        "immediate_advantage_dqn": "lambda_0p3_immediate",
        "double_dqn": "double",
        "double_centered_full_action_dqn": "double_lambda_0p3",
        "contextual_bandit": "gamma_0",
    }[variant]
    return results / "models" / stage / f"{variant}__{configuration}__seed_{seed}.pth"


def curve_path(path: Path) -> Path:
    return path.with_name(path.stem + "__curve.csv")


def save_checkpoint(path: Path, agent: DQNAgent, seed: int, episodes: int,
                    horizon: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "algorithm": "pure_standard_dqn_backbone",
        "variant": agent.variant,
        "model_seed": seed,
        "episodes": episodes,
        "horizon": horizon,
        "state_dim": SatelliteSchedulingEnv.state_dim,
        "action_dim": 3,
        "agent_config": asdict(agent.config),
        "bellman_target": (
            "online_argmax_target_evaluation"
            if agent.uses_double_target
            else "target_network_max_not_double_dqn"
        ),
        "replay": "uniform_one_step",
        "network": f"{SatelliteSchedulingEnv.state_dim}-128-128-64-3",
        "online_state_dict": agent.online.state_dict(),
        "target_state_dict": agent.target.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "training_steps": agent.steps,
        "epsilon": agent.epsilon,
        "torch_version": torch.__version__,
        "training_accounting": getattr(agent, "training_accounting", None),
    }
    torch.save(payload, path)


def _validate_legacy_config(payload: dict, expected: AgentConfig, variant: str) -> None:
    actual = payload.get("agent_config", {})
    shared = (
        "gamma", "lr", "batch_size", "buffer_size", "target_interval",
        "epsilon_start", "epsilon_end", "epsilon_decay_steps",
    )
    for key in shared:
        if key in actual and actual[key] != getattr(expected, key):
            raise ValueError(f"checkpoint config mismatch: {key}")
    if DQNAgent._variant_uses_counterfactuals(variant):
        weight = actual.get("advantage_lambda", actual.get("auxiliary_lambda"))
    else:
        weight = 0.0
    if DQNAgent._variant_uses_counterfactuals(variant) and weight != expected.auxiliary_lambda:
        raise ValueError(f"checkpoint auxiliary weight mismatch: {weight}")


def load_checkpoint(path: Path, variant: str, seed: int, device: torch.device,
                    episodes: int = EPISODES, horizon: int = HORIZON,
                    auxiliary_lambda: float = AUXILIARY_LAMBDA) -> DQNAgent:
    config = config_for(variant, episodes, horizon, auxiliary_lambda)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("variant", payload.get("policy")) != variant:
        raise ValueError(f"checkpoint variant mismatch: {path}")
    if int(payload.get("model_seed", -1)) != seed:
        raise ValueError(f"checkpoint seed mismatch: {path}")
    if int(payload.get("episodes", episodes)) != episodes:
        raise ValueError(f"checkpoint episode mismatch: {path}")
    if int(payload.get("horizon", horizon)) != horizon:
        raise ValueError(f"checkpoint horizon mismatch: {path}")
    if int(payload.get("state_dim", -1)) != SatelliteSchedulingEnv.state_dim:
        raise ValueError(
            f"checkpoint state dimension mismatch: {path}; retraining is required"
        )
    _validate_legacy_config(payload, config, variant)
    agent = DQNAgent(
        variant, config, device, seed, state_dim=SatelliteSchedulingEnv.state_dim
    )
    agent.online.load_state_dict(payload["online_state_dict"])
    agent.target.load_state_dict(payload["target_state_dict"])
    if "optimizer_state_dict" in payload:
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.steps = int(payload.get("training_steps", episodes * horizon))
    agent.epsilon = float(payload.get("epsilon", config.epsilon_end))
    agent.training_accounting = payload.get("training_accounting")
    return agent


def train_model(variant: str, seed: int, episodes: int, horizon: int,
                device: torch.device, auxiliary_lambda: float = AUXILIARY_LAMBDA,
                environment_config: EnvConfig | None = None,
                preview_transition_scale: float = 1.0,
                ) -> tuple[DQNAgent, list[dict]]:
    set_deterministic(seed)
    started = time.perf_counter()
    config = config_for(variant, episodes, horizon, auxiliary_lambda)
    env = SatelliteSchedulingEnv(
        environment_config
        if environment_config is not None
        else EnvConfig(horizon=horizon, scenario="nominal")
    )
    if env.config.horizon != horizon:
        raise ValueError("training environment horizon does not match budget horizon")
    agent = DQNAgent(
        variant, config, device, seed, state_dim=env.state_dim,
        action_dim=env.action_dim,
    )
    preview_calls = 0
    preview_action_evaluations = 0
    optimizer_updates = 0
    curve: list[dict] = []
    recent: list[float] = []
    for episode in range(episodes):
        state = env.reset(seed=seed * 1_000_000 + episode)
        total_cost = 0.0
        episode_metrics = {
            key: 0.0 for key in (
                "immediate_cost", "penalty", "energy_use", "latency_ms",
                "thermal_violation", "energy_violation", "queue_violation",
                "contact_violation",
            )
        }
        updates: list[dict[str, float]] = []
        done = False
        while not done:
            if agent.uses_counterfactuals:
                preview_costs, preview_resources = env.preview_all_actions()
                preview_calls += 1
                preview_action_evaluations += env.action_dim
                if preview_transition_scale != 1.0:
                    preview_costs = preview_costs * preview_transition_scale
                    current_resources = state[-env.resource_dim:]
                    preview_resources = current_resources + preview_transition_scale * (
                        preview_resources - current_resources
                    )
                    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.08, 0.05])
                    upper = np.array([1.3, 1.0, 1.3, 1.2, 1.0, 1.0])
                    preview_resources = np.clip(preview_resources, lower, upper)
            else:
                preview_costs = preview_resources = None
            action = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.observe(
                state, action, reward, next_state, done,
                preview_costs, preview_resources,
            )
            if agent.steps % 4 == 0:
                update = agent.update()
                if update is not None:
                    updates.append(update)
                    optimizer_updates += 1
            total_cost += info["total_cost"]
            for key in episode_metrics:
                episode_metrics[key] += info[key]
            state = next_state
        recent.append(total_cost)
        curve.append(
            {
                "variant": variant,
                "model_seed": seed,
                "episode": episode,
                "cost": total_cost,
                "rolling_cost": float(np.mean(recent[-20:])),
                **episode_metrics,
                "epsilon": agent.epsilon,
                "loss": float(np.mean([x["loss"] for x in updates]))
                if updates else np.nan,
                "td_loss": float(np.mean([x["td_loss"] for x in updates]))
                if updates else np.nan,
                "auxiliary_loss": float(
                    np.mean([x["auxiliary_loss"] for x in updates])
                ) if updates else np.nan,
            }
        )
    if agent.steps != episodes * horizon:
        raise AssertionError("training interaction budget changed")
    agent.training_accounting = {
        "wall_time_s": time.perf_counter() - started,
        "real_interactions": agent.steps,
        "optimizer_updates": optimizer_updates,
        "preview_calls": preview_calls,
        "preview_action_evaluations": preview_action_evaluations,
    }
    return agent, curve


def train_or_load(results: Path, stage: str, variant: str, seed: int,
                  episodes: int, horizon: int, device: torch.device,
                  resume: bool = True, auxiliary_lambda: float = AUXILIARY_LAMBDA
                  ) -> DQNAgent:
    path = model_path(results, stage, variant, seed)
    if resume and path.exists():
        return load_checkpoint(
            path, variant, seed, device, episodes, horizon, auxiliary_lambda
        )
    agent, curve = train_model(
        variant, seed, episodes, horizon, device, auxiliary_lambda
    )
    save_checkpoint(path, agent, seed, episodes, horizon)
    pd.DataFrame(curve).to_csv(curve_path(path), index=False)
    return agent


def threshold_action(state: np.ndarray, costs: np.ndarray) -> int:
    heat, energy, queue, _, bandwidth, contact = state[-6:]
    if heat > 0.72 or energy < 0.24:
        return 1 if bandwidth > 0.28 and contact > 0.18 else 2
    if bandwidth < 0.22 or contact < 0.14:
        return 0
    if queue > 0.70:
        return 2
    return int(np.argmin(costs))


def evaluate_policy(policy: str, agent: DQNAgent | None, model_seed: int,
                    trace_seeds: Iterable[int], horizon: int, scenario_label: str,
                    environment_scenario: str, coupling: float) -> list[dict]:
    rows: list[dict] = []
    for trace_seed in trace_seeds:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=horizon, coupling=coupling, scenario=environment_scenario)
        )
        state = env.reset(seed=trace_seed)
        totals = {
            key: 0.0 for key in (
                "total_cost", "immediate_cost", "penalty", "thermal_violation",
                "energy_violation", "queue_violation", "contact_violation",
            )
        }
        actions = np.zeros(3, dtype=int)
        done = False
        while not done:
            costs = env.immediate_costs()
            if policy == "immediate_argmin":
                action = int(np.argmin(costs))
            elif policy == "threshold":
                action = threshold_action(state, costs)
            elif policy == "fixed_onboard":
                action = 0
            elif policy == "fixed_ground":
                action = 1
            else:
                if agent is None:
                    raise ValueError(f"missing agent for {policy}")
                action = agent.act(state, explore=False)
            state, _, done, info = env.step(action)
            actions[action] += 1
            for key in totals:
                totals[key] += info[key]
        rows.append(
            {
                "policy": policy,
                "model_seed": model_seed,
                "trace_seed": trace_seed,
                "scenario": scenario_label,
                "environment_scenario": environment_scenario,
                "coupling": coupling,
                **totals,
                "onboard_fraction": actions[0] / horizon,
                "ground_fraction": actions[1] / horizon,
                "hybrid_fraction": actions[2] / horizon,
            }
        )
    return rows


METRICS = (
    "total_cost", "immediate_cost", "penalty", "thermal_violation",
    "energy_violation", "queue_violation", "contact_violation",
    "onboard_fraction", "ground_fraction", "hybrid_fraction",
)


def seed_level(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(
        ["scenario", "coupling", "policy", "model_seed"], as_index=False
    )[list(METRICS)].mean()


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    by_seed = seed_level(raw)
    grouped = by_seed.groupby(["scenario", "coupling", "policy"])[list(METRICS)]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std(ddof=1).add_suffix("_sd")
    count = grouped.size().rename("n_model_seeds")
    return pd.concat([mean, sd, count], axis=1).reset_index()


def paired_bootstrap(raw: pd.DataFrame) -> pd.DataFrame:
    by_seed = seed_level(raw)
    core_comparisons = (
        ("centered_full_action_dqn", "standard_dqn", "complete centered full-action"),
    )
    extra_comparisons = (
        ("centered_full_action_dqn", "immediate_argmin", "long-term learned scheduling"),
        ("centered_full_action_dqn", "contextual_bandit", "long-term Bellman term"),
        ("centered_full_action_dqn", "threshold", "learned versus threshold"),
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    # Process the five locked regimes first and in their original order so the
    # first three comparisons reproduce the archived bootstrap draws exactly.
    controls = tuple(item for item in SCENARIOS if item[2] != 1.0)
    scenario_order = FULL_COUPLING + controls
    for comparison_group in (core_comparisons, extra_comparisons):
        for label, _, coupling in scenario_order:
            pivot = by_seed[
                (by_seed.scenario == label) & (by_seed.coupling == coupling)
            ].pivot(index="model_seed", columns="policy", values="total_cost")
            for treatment, reference, interpretation in comparison_group:
                difference = (pivot[treatment] - pivot[reference]).dropna().to_numpy()
                boot = rng.choice(
                    difference, size=(BOOTSTRAP_RESAMPLES, len(difference)), replace=True
                ).mean(axis=1)
                low = float(np.quantile(boot, 0.025))
                high = float(np.quantile(boot, 0.975))
                reference_mean = float(pivot[reference].mean())
                rows.append(
                    {
                        "scenario": label,
                        "comparison": f"{treatment} - {reference}",
                        "interpretation": interpretation,
                        "n_pairs": len(difference),
                        "mean_difference": float(difference.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "relative_improvement_percent": float(
                            -100.0 * difference.mean() / reference_mean
                        ),
                        "mean_better": bool(difference.mean() < 0),
                        "significant": bool(high < 0),
                    }
                )
    return pd.DataFrame(rows)


def write_manifest(results: Path) -> None:
    rows: list[dict] = []
    for path in sorted((results / "models").rglob("*.pth")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "file": path.relative_to(results).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "variant": payload.get("variant", payload.get("policy")),
                "model_seed": payload.get("model_seed"),
                "episodes": payload.get("episodes"),
                "horizon": payload.get("horizon"),
                "training_steps": payload.get("training_steps"),
            }
        )
    pd.DataFrame(rows).to_csv(results / "model_manifest.csv", index=False)


def run_evaluation(results: Path, scenarios: tuple, episodes: int, horizon: int,
                   test_traces: int, device: torch.device, resume: bool) -> pd.DataFrame:
    agents: dict[tuple[str, int], DQNAgent] = {}
    for seed in MODEL_SEEDS:
        for variant in LEARNED:
            stage = "contextual_bandit" if variant == "contextual_bandit" else "confirmatory"
            agents[(variant, seed)] = train_or_load(
                results, stage, variant, seed, episodes, horizon, device, resume
            )
    rows: list[dict] = []
    for seed_index, seed in enumerate(MODEL_SEEDS):
        traces = [TRACE_BASE + seed_index * 1_000 + i for i in range(test_traces)]
        for label, environment_scenario, coupling in scenarios:
            for policy in ALL_POLICIES:
                rows.extend(
                    evaluate_policy(
                        policy,
                        agents.get((policy, seed)),
                        seed,
                        traces,
                        horizon,
                        label,
                        environment_scenario,
                        coupling,
                    )
                )
    return pd.DataFrame(rows)


def write_evaluation(results: Path, raw: pd.DataFrame, stem: str) -> None:
    results.mkdir(parents=True, exist_ok=True)
    raw.to_csv(results / f"{stem}_raw.csv", index=False)
    seed_level(raw).to_csv(results / f"{stem}_seed_level.csv", index=False)
    summarize(raw).to_csv(results / f"{stem}_summary.csv", index=False)
    paired_bootstrap(raw).to_csv(results / f"{stem}_paired_bootstrap.csv", index=False)
    metadata = {
        "episodes": EPISODES,
        "horizon": HORIZON,
        "real_steps_per_model": EPISODES * HORIZON,
        "model_seeds": list(MODEL_SEEDS),
        "test_traces_per_seed": int(raw.trace_seed.nunique() / len(MODEL_SEEDS)),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "auxiliary_lambda": AUXILIARY_LAMBDA,
        "bellman_target": "standard target-network max; not Double DQN",
        "replay": "uniform one-step",
    }
    (results / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_manifest(results)


def calibration(results: Path, episodes: int, horizon: int, device: torch.device,
                resume: bool) -> None:
    grid = (0.1, 0.3, 1.0, 3.0)
    rows: list[dict] = []
    for weight in grid:
        for seed in CALIBRATION_SEEDS:
            # Calibration artifacts remain reproducible but are never consulted
            # by seven-scenario confirmation after lambda=0.3 is locked.
            variant = "centered_full_action_dqn"
            path = results / "models" / "calibration" / (
                f"{variant}__lambda_{str(weight).replace('.', 'p')}__seed_{seed}.pth"
            )
            if resume and path.exists():
                agent = load_checkpoint(
                    path, variant, seed, device, episodes, horizon, weight
                )
            else:
                agent, curve = train_model(
                    variant, seed, episodes, horizon, device, weight
                )
                save_checkpoint(path, agent, seed, episodes, horizon)
                pd.DataFrame(curve).to_csv(curve_path(path), index=False)
            rows.append({"weight": weight, "model_seed": seed, "path": path.as_posix()})
    pd.DataFrame(rows).to_csv(results / "calibration_registry.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("calibration", "confirmatory", "seven-scenarios", "all"),
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--test-traces", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.episodes, args.horizon, args.test_traces = 12, 16, 3
    args.results.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"stage={args.stage} device={device} torch={torch.__version__}", flush=True)
    if args.stage == "calibration":
        calibration(args.results, args.episodes, args.horizon, device, args.resume)
        return
    if args.stage == "confirmatory":
        raw = run_evaluation(
            args.results, FULL_COUPLING, args.episodes, args.horizon,
            args.test_traces, device, args.resume,
        )
        write_evaluation(args.results, raw, args.stage)
        return
    if args.stage == "all" and not args.smoke:
        # Existing locked calibration assets are retained; all intentionally
        # avoids data-dependent retuning on confirmatory seeds.
        registry = args.results / "calibration_registry.csv"
        if not registry.exists():
            print("locked calibration registry absent; confirmation remains lambda=0.3")
    raw = run_evaluation(
        args.results, SCENARIOS, args.episodes, args.horizon,
        args.test_traces, device, args.resume,
    )
    expected = len(SCENARIOS) * len(ALL_POLICIES) * len(MODEL_SEEDS) * args.test_traces
    if len(raw) != expected:
        raise AssertionError(f"expected {expected} rows, found {len(raw)}")
    write_evaluation(args.results, raw, "seven_scenarios")
    print(f"wrote {len(raw)} evaluation rows to {args.results}", flush=True)


if __name__ == "__main__":
    main()
