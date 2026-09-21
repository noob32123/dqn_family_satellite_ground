"""Second-round reviewer experiments.

The module writes to a new results directory and never changes the locked
reviewer-revision artifacts.  It covers discount-factor sensitivity, the
lightweight greedy rollout comparator, matched CPU decision-time measurement,
and training-length checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import copy
import platform

import numpy as np
import pandas as pd
import torch

from .agent import DQNAgent
from .environment import SatelliteSchedulingEnv
from .experiment import (
    AUXILIARY_LAMBDA,
    HORIZON,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    config_for,
    curve_path,
    save_checkpoint,
    set_deterministic,
    train_model,
)
from .planner import PlannerConfig, RecedingHorizonPlanner
from .reviewer_experiments import (
    TRACE_BASE,
    _exact_two_sided_sign_p,
    _holm_adjust,
    revision_env,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "revision_round2"
SEEDS = tuple(range(800, 820))
VARIANTS = (
    "standard_dqn",
    "full_action_q_dqn",
    "immediate_advantage_dqn",
    "centered_full_action_dqn",
    "double_dqn",
    "double_centered_full_action_dqn",
)
GAMMAS = (0.90, 0.95, 0.97, 0.99)
TRACE_COUNT = 20
TRAINING_LENGTHS = (300, 600, 900)
FIXED_EPSILON_DECAY_STEPS = 13_440


def trace_seed_for(seed: int, index: int) -> int:
    """Use exactly the original 800--819 nominal evaluation namespaces."""
    return TRACE_BASE + (seed - 800) * 1000 + index


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def model_path(results: Path, variant: str, gamma: float, seed: int) -> Path:
    return results / "gamma_models" / f"{variant}__gamma_{tag(gamma)}__seed_{seed}.pth"


def train_gamma(results: Path, gamma: float, seeds: tuple[int, ...], episodes: int,
                horizon: int, device: torch.device, resume: bool,
                variants: tuple[str, ...] = VARIANTS) -> pd.DataFrame:
    rows = []
    for variant in variants:
        for seed in seeds:
            path = model_path(results, variant, gamma, seed)
            curve_file = curve_path(path)
            locked_tag = {
                "standard_dqn": "standard",
                "full_action_q_dqn": "lambda_0p3_uncentered",
                "immediate_advantage_dqn": "lambda_0p3_immediate",
                "centered_full_action_dqn": "lambda_0p3",
                "double_dqn": "double",
                "double_centered_full_action_dqn": "double_lambda_0p3",
            }[variant]
            locked = (ROOT / "results" / "reviewer_revision_state_complete" / "models" /
                      f"{variant}__{locked_tag}__seed_{seed}.pth")
            locked_curve = curve_path(locked)
            if (resume and gamma == 0.97 and episodes == 600 and not path.exists() and locked.exists()
                    and locked_curve.exists()):
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(locked, path)
                shutil.copy2(locked_curve, curve_file)
            if resume and path.exists() and curve_file.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                curve = pd.read_csv(curve_file)
            else:
                print(f"training gamma={gamma} variant={variant} seed={seed}", flush=True)
                agent, curve_rows = train_model(
                    variant, seed, episodes, horizon, device,
                    AUXILIARY_LAMBDA,
                    environment_config=revision_env(horizon),
                    gamma_override=gamma,
                )
                save_checkpoint(path, agent, seed, episodes, horizon)
                curve = pd.DataFrame(curve_rows)
                curve.to_csv(curve_file, index=False)
                payload = torch.load(path, map_location="cpu", weights_only=False)
            rows.append({
                "variant": variant,
                "gamma": gamma,
                "model_seed": seed,
                "episodes": episodes,
                "training_steps": int(payload["training_steps"]),
                "curve_file": curve_file.as_posix(),
                "final_cost": float(curve.iloc[-1]["cost"]),
                "final_rolling_cost": float(curve.iloc[-1]["rolling_cost"]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(results / f"gamma_{tag(gamma)}_manifest.csv", index=False)
    return frame


def load_gamma_agent(results: Path, variant: str, gamma: float, seed: int,
                     episodes: int, horizon: int, device: torch.device,
                     epsilon_decay_steps_override: int | None = None,
                     path_override: Path | None = None) -> DQNAgent:
    path = path_override or model_path(results, variant, gamma, seed)
    payload = torch.load(path, map_location=device, weights_only=False)
    config = config_for(
        variant, episodes, horizon, AUXILIARY_LAMBDA, gamma,
        epsilon_decay_steps_override,
    )
    agent = DQNAgent(variant, config, device, seed,
                     state_dim=SatelliteSchedulingEnv.state_dim)
    agent.online.load_state_dict(payload["online_state_dict"])
    agent.target.load_state_dict(payload["target_state_dict"])
    agent.steps = int(payload["training_steps"])
    agent.epsilon = float(payload.get("epsilon", config.epsilon_end))
    return agent


def length_model_path(results: Path, variant: str, seed: int,
                      checkpoint: int) -> Path:
    return (results / "training_length_models" /
            f"{variant}__gamma_0p97__seed_{seed}__checkpoint_{checkpoint}.pth")


def train_length(results: Path, seeds: tuple[int, ...], horizon: int,
                 device: torch.device, resume: bool,
                 variants: tuple[str, ...] = VARIANTS) -> pd.DataFrame:
    """Train each objective for 900 episodes and persist 300/600/900 snapshots."""
    rows = []
    for variant in variants:
        for seed in seeds:
            paths = {n: length_model_path(results, variant, seed, n)
                     for n in TRAINING_LENGTHS}
            curve_paths = {n: curve_path(p) for n, p in paths.items()}
            if resume and all(p.exists() and q.exists()
                              for p, q in zip(paths.values(), curve_paths.values())):
                for n in TRAINING_LENGTHS:
                    payload = torch.load(paths[n], map_location="cpu", weights_only=False)
                    rows.append({
                        "variant": variant, "model_seed": seed,
                        "checkpoint_episode": n,
                        "training_steps": int(payload["training_steps"]),
                        "curve_file": curve_paths[n].as_posix(),
                    })
                continue
            print(f"training length variant={variant} seed={seed}", flush=True)

            def checkpoint_callback(agent: DQNAgent, curve: list[dict], episode: int) -> None:
                if episode not in TRAINING_LENGTHS:
                    return
                target = paths[episode]
                save_checkpoint(target, agent, seed, 900, horizon)
                payload = torch.load(target, map_location="cpu", weights_only=False)
                payload["checkpoint_episode"] = episode
                payload["epsilon_decay_steps_fixed"] = FIXED_EPSILON_DECAY_STEPS
                torch.save(payload, target)
                target_curve = pd.DataFrame(curve)
                target_curve.to_csv(curve_paths[episode], index=False)

            train_model(
                variant, seed, 900, horizon, device, AUXILIARY_LAMBDA,
                environment_config=revision_env(horizon), gamma_override=0.97,
                epsilon_decay_steps_override=FIXED_EPSILON_DECAY_STEPS,
                episode_callback=checkpoint_callback,
            )
            for n in TRAINING_LENGTHS:
                payload = torch.load(paths[n], map_location="cpu", weights_only=False)
                rows.append({
                    "variant": variant, "model_seed": seed,
                    "checkpoint_episode": n,
                    "training_steps": int(payload["training_steps"]),
                    "curve_file": curve_paths[n].as_posix(),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(results / "training_length_manifest.csv", index=False)
    return frame


def evaluate_training_length(results: Path, seeds: tuple[int, ...], horizon: int,
                             traces: int, device: torch.device) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            for checkpoint in TRAINING_LENGTHS:
                agent = load_gamma_agent(
                    results, variant, 0.97, seed, 900, horizon, device,
                    epsilon_decay_steps_override=FIXED_EPSILON_DECAY_STEPS,
                    path_override=length_model_path(results, variant, seed, checkpoint),
                )
                for j in range(traces):
                    result = evaluate_action_policy(
                        variant, agent, trace_seed_for(seed, j),
                        revision_env(horizon), False,
                    )
                    rows.append({
                        "variant": variant, "model_seed": seed,
                        "checkpoint_episode": checkpoint, **result,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(results / "training_length_evaluation.csv", index=False)
    return frame


def greedy_rollout_action(env: SatelliteSchedulingEnv, horizon: int = 8,
                          gamma: float = 0.97) -> int:
    """Choose the first action on a deterministic greedy predicted rollout."""
    depth = min(horizon, env.config.horizon - env.t)
    best = None
    for first in range(env.action_dim):
        resources = env.resources.copy()
        score = 0.0
        for offset in range(depth):
            if offset == 0:
                action = first
            else:
                candidates = []
                for action_candidate in range(env.action_dim):
                    next_resources, info = env.transition_from(
                        env.t + offset, resources, action_candidate
                    )
                    candidates.append((float(info["total_cost"]), action_candidate,
                                       next_resources, info))
                _, action, next_resources, info = min(candidates, key=lambda x: (x[0], x[1]))
            if offset == 0:
                next_resources, info = env.transition_from(env.t, resources, action)
            score += gamma ** offset * float(info["total_cost"])
            resources = next_resources
        candidate = (score, first)
        if best is None or candidate < best:
            best = candidate
    return int(best[1])


def evaluate_action_policy(policy: str, agent: DQNAgent | None, trace_seed: int,
                           config, timing: bool = False, gamma: float = 0.97) -> dict:
    env = SatelliteSchedulingEnv(config)
    state = env.reset(seed=trace_seed)
    total = 0.0
    action_times = []
    done = False
    planner = RecedingHorizonPlanner(PlannerConfig(4, gamma))
    while not done:
        started = time.perf_counter() if timing else None
        if policy == "greedy8":
            action = greedy_rollout_action(env, 8, gamma)
        elif policy == "mpc4":
            action = planner.act(env)
        else:
            action = agent.act(state, explore=False)
        if started is not None:
            action_times.append(1000.0 * (time.perf_counter() - started))
        state, _, done, info = env.step(action)
        total += float(info["total_cost"])
    result = {"total_cost": total, "trace_seed": trace_seed}
    if timing:
        result.update({
            "decision_ms_mean": float(np.mean(action_times)),
            "decision_ms_median": float(np.median(action_times)),
            "decision_ms_p95": float(np.quantile(action_times, 0.95)),
            "decision_count": len(action_times),
        })
    return result


def evaluate_gamma(results: Path, gamma: float, seeds: tuple[int, ...], episodes: int,
                    horizon: int, traces: int, device: torch.device) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            agent = load_gamma_agent(results, variant, gamma, seed, episodes, horizon, device)
            for j in range(traces):
                row = evaluate_action_policy(
                    variant, agent, trace_seed_for(seed, j),
                    revision_env(horizon), False,
                )
                rows.append({"variant": variant, "gamma": gamma, "model_seed": seed, **row})
    frame = pd.DataFrame(rows)
    frame.to_csv(results / f"gamma_{tag(gamma)}_evaluation.csv", index=False)
    return frame


def evaluate_baselines(results: Path, seeds: tuple[int, ...], horizon: int,
                       traces: int, resume: bool = False) -> pd.DataFrame:
    """Evaluate forecast-enabled baselines on the same 20 x 20 traces as DQN."""
    rows = []
    for gamma in GAMMAS:
        target = results / f"mpc_gamma_{tag(gamma)}_evaluation.csv"
        if resume and target.exists():
            rows.extend(pd.read_csv(target).to_dict("records"))
            continue
        block = []
        for seed in seeds:
            print(f"evaluating mpc4 gamma={gamma} seed={seed}", flush=True)
            for j in range(traces):
                result = evaluate_action_policy(
                    "mpc4", None, trace_seed_for(seed, j),
                    revision_env(horizon), gamma=gamma,
                )
                block.append({"policy": "mpc4", "gamma": gamma,
                              "model_seed": seed, **result})
        pd.DataFrame(block).to_csv(target, index=False)
        rows.extend(block)
    target = results / "greedy8_evaluation.csv"
    if resume and target.exists():
        block = pd.read_csv(target).to_dict("records")
    else:
        block = []
        for seed in seeds:
            print(f"evaluating greedy8 seed={seed}", flush=True)
            for j in range(traces):
                result = evaluate_action_policy(
                    "greedy8", None, trace_seed_for(seed, j), revision_env(horizon),
                )
                block.append({"policy": "greedy8", "gamma": 0.97,
                              "model_seed": seed, **result})
        pd.DataFrame(block).to_csv(target, index=False)
    rows.extend(block)
    frame = pd.DataFrame(rows)
    frame.to_csv(results / "baseline_evaluation.csv", index=False)
    return frame


def runtime_benchmark(results: Path, seeds: tuple[int, ...], horizon: int,
                      device: torch.device) -> pd.DataFrame:
    """Three repetitions on identical saved states, batch=1 and CPU thread=1.

    States come from the locked standard-DQN trajectory, one nominal trace per
    seed and every fourth decision (including the final decision). Setup, model
    loading, and environment transitions are outside the measured interval.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cpu = torch.device("cpu")
    samples = []
    agents = {}
    for seed in seeds:
        for variant in VARIANTS:
            agents[variant, seed] = load_gamma_agent(
                results, variant, 0.97, seed, 600, horizon, cpu,
            )
            agents[variant, seed].online.eval()
        env = SatelliteSchedulingEnv(revision_env(horizon))
        state = env.reset(seed=trace_seed_for(seed, 0))
        for step in range(horizon):
            if step % 4 == 0 or step == horizon - 1:
                samples.append((seed, step, state.copy(), copy.deepcopy(env)))
            action = agents["standard_dqn", seed].act(state, explore=False)
            state, _, _, _ = env.step(action)
    planner = RecedingHorizonPlanner(PlannerConfig(4, 0.97))
    policies = (*VARIANTS, "mpc4", "greedy8")

    def act(policy, seed, state, env):
        if policy == "mpc4":
            return planner.act(env)
        if policy == "greedy8":
            return greedy_rollout_action(env)
        return agents[policy, seed].act(state, explore=False)

    for policy in policies:
        for seed, step, state, env in samples[:20]:
            act(policy, seed, state, env)
    rows = []
    for repeat in range(3):
        # Rotate order between repetitions to reduce systematic order effects.
        ordered = policies[repeat:] + policies[:repeat]
        for policy in ordered:
            for sample_id, (seed, step, state, env) in enumerate(samples):
                started = time.perf_counter_ns()
                action = act(policy, seed, state, env)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                rows.append({
                    "policy": policy, "repeat": repeat, "sample_id": sample_id,
                    "model_seed": seed, "trace_seed": trace_seed_for(seed, 0),
                    "step": step, "action": action, "decision_ms": elapsed,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(results / "runtime_benchmark.csv", index=False)
    pd.DataFrame([
        dict(policy=policy, n_decisions=len(group),
             decision_ms_mean=group.decision_ms.mean(),
             decision_ms_median=group.decision_ms.median(),
             decision_ms_p95=group.decision_ms.quantile(0.95))
        for policy, group in frame.groupby("policy")
    ]).to_csv(results / "runtime_benchmark_summary.csv", index=False)
    np.savez_compressed(
        results / "runtime_states.npz",
        states=np.stack([x[2] for x in samples]),
        model_seeds=np.array([x[0] for x in samples]),
        steps=np.array([x[1] for x in samples]),
    )
    (results / "runtime_metadata.json").write_text(json.dumps({
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "batch_size": 1, "repetitions": 3, "shared_states": len(samples),
        "warmup_decisions_per_policy": min(20, len(samples)),
        "state_source": "standard DQN, trace index 0, every fourth step plus terminal",
    }, indent=2), encoding="utf-8")
    return frame


def paired_inference(frame: pd.DataFrame, group_columns: list[str],
                     treatment_column: str, reference_value,
                     output: Path) -> pd.DataFrame:
    """Paired seed-block bootstrap and Holm-adjusted exact sign tests."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for keys, selected in frame.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        index_columns = ["model_seed"]
        pivot = selected.pivot(index=index_columns, columns=treatment_column,
                                values="total_cost")
        if reference_value not in pivot.columns:
            continue
        for treatment in pivot.columns:
            if treatment == reference_value:
                continue
            difference = (pivot[treatment] - pivot[reference_value]).dropna().to_numpy()
            boot = rng.choice(difference, size=(BOOTSTRAP_RESAMPLES, len(difference)),
                              replace=True).mean(axis=1)
            row = dict(zip(group_columns, keys))
            row.update({
                "treatment": treatment,
                "reference": reference_value,
                "n_pairs": len(difference),
                "mean_difference": float(difference.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "sign_p": _exact_two_sided_sign_p(difference),
            })
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result):
        result["holm_p"] = _holm_adjust(result.sign_p.to_numpy())
    result.to_csv(output, index=False)
    return result


def run_inference(results: Path) -> None:
    gamma_files = sorted(results.glob("gamma_*_evaluation.csv"))
    if gamma_files:
        gamma = pd.concat([pd.read_csv(path) for path in gamma_files], ignore_index=True)
        gamma_seed = gamma.groupby(
            ["variant", "gamma", "model_seed"], as_index=False
        ).total_cost.mean()
        paired_inference(gamma_seed, ["variant"], "gamma", 0.97,
                         results / "gamma_sensitivity_inference.csv")

        baseline = results / "baseline_evaluation.csv"
        if baseline.exists():
            mpc = pd.read_csv(baseline)
            mpc = mpc[mpc.policy == "mpc4"].rename(columns={"policy": "variant"})
            mpc_seed = mpc.groupby(
                ["variant", "gamma", "model_seed"], as_index=False
            ).total_cost.mean()
            joint = pd.concat([gamma_seed, mpc_seed], ignore_index=True)
            paired_inference(joint, ["gamma"], "variant", "mpc4",
                             results / "dqn_vs_mpc_gamma_inference.csv")

            greedy = pd.read_csv(baseline)
            greedy = greedy[greedy.policy == "greedy8"].rename(
                columns={"policy": "variant"}
            )
            greedy_seed = greedy.groupby(
                ["variant", "gamma", "model_seed"], as_index=False
            ).total_cost.mean()
            nominal = pd.concat(
                [gamma_seed[gamma_seed.gamma == 0.97], greedy_seed], ignore_index=True
            )
            paired_inference(nominal, ["gamma"], "variant", "greedy8",
                             results / "dqn_vs_greedy8_inference.csv")

    length_file = results / "training_length_evaluation.csv"
    if length_file.exists():
        length = pd.read_csv(length_file).groupby(
            ["variant", "checkpoint_episode", "model_seed"], as_index=False
        ).total_cost.mean()
        paired_inference(length, ["variant"], "checkpoint_episode", 600,
                         results / "training_length_inference.csv")


def summarize(results: Path) -> None:
    evaluations = []
    for path in sorted(results.glob("gamma_*_evaluation.csv")):
        evaluations.append(pd.read_csv(path))
    if evaluations:
        all_eval = pd.concat(evaluations, ignore_index=True)
        by_seed = all_eval.groupby(["gamma", "variant", "model_seed"], as_index=False).total_cost.mean()
        summary = by_seed.groupby(["gamma", "variant"]).total_cost.agg(["mean", "std", "count"]).reset_index()
        summary.to_csv(results / "gamma_sensitivity_summary.csv", index=False)
    length_eval = results / "training_length_evaluation.csv"
    if length_eval.exists():
        frame = pd.read_csv(length_eval)
        by_seed = frame.groupby(
            ["variant", "checkpoint_episode", "model_seed"], as_index=False
        ).total_cost.mean()
        by_seed.groupby(["variant", "checkpoint_episode"]).total_cost.agg(
            ["mean", "std", "count"]
        ).reset_index().to_csv(results / "training_length_summary.csv", index=False)
    baseline = results / "baseline_evaluation.csv"
    if baseline.exists():
        frame = pd.read_csv(baseline)
        by_seed = frame.groupby(
            ["policy", "gamma", "model_seed"], as_index=False
        ).total_cost.mean()
        by_seed.groupby(["policy", "gamma"]).total_cost.agg(
            ["mean", "std", "count"]
        ).reset_index().to_csv(results / "baseline_summary.csv", index=False)
    run_inference(results)


def validate(results: Path) -> None:
    report = {"gamma_values": list(GAMMAS), "variants": list(VARIANTS),
              "trace_count_per_seed": TRACE_COUNT,
              "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
              "bootstrap_seed": BOOTSTRAP_SEED}
    manifests = sorted(results.glob("gamma_*_manifest.csv"))
    if len(manifests) != len(GAMMAS):
        raise AssertionError(f"expected {len(GAMMAS)} gamma manifests, found {len(manifests)}")
    for path in manifests:
        frame = pd.read_csv(path)
        expected = len(VARIANTS) * frame.model_seed.nunique()
        if len(frame) != expected:
            raise AssertionError(f"{path.name} has {len(frame)} rows, expected {expected}")
        if not (frame.training_steps == frame.episodes * HORIZON).all():
            raise AssertionError(f"training budget mismatch in {path.name}")
    for path in results.glob("gamma_*_evaluation.csv"):
        frame = pd.read_csv(path)
        per_cell = frame.groupby(["variant", "model_seed"]).size()
        expected = len(VARIANTS) * frame.model_seed.nunique() * int(per_cell.iloc[0])
        if len(frame) != expected:
            raise AssertionError(f"evaluation row mismatch in {path.name}")
        if not np.isfinite(frame.total_cost).all():
            raise AssertionError(f"non-finite result in {path.name}")
    length_manifest = results / "training_length_manifest.csv"
    if length_manifest.exists():
        frame = pd.read_csv(length_manifest)
        expected = len(VARIANTS) * frame.model_seed.nunique() * len(TRAINING_LENGTHS)
        if len(frame) != expected:
            raise AssertionError(f"training length manifest has {len(frame)} rows, expected {expected}")
        if not (frame.training_steps == frame.checkpoint_episode * HORIZON).all():
            raise AssertionError("training length checkpoint budget mismatch")
    length_eval = results / "training_length_evaluation.csv"
    if length_eval.exists():
        frame = pd.read_csv(length_eval)
        expected = (len(VARIANTS) * frame.model_seed.nunique() *
                    len(TRAINING_LENGTHS) * TRACE_COUNT)
        if len(frame) != expected:
            raise AssertionError(f"training length evaluation has {len(frame)} rows, expected {expected}")
        if not np.isfinite(frame.total_cost).all():
            raise AssertionError("non-finite training length result")
        gamma_reference = results / "gamma_0p97_evaluation.csv"
        if gamma_reference.exists():
            reference = pd.read_csv(gamma_reference)
            checkpoint = frame[frame.checkpoint_episode == 600]
            merged = checkpoint.merge(
                reference,
                on=["variant", "model_seed", "trace_seed"],
                suffixes=("_length", "_gamma"),
                validate="one_to_one",
            )
            if len(merged) != len(reference) or not np.allclose(
                merged.total_cost_length, merged.total_cost_gamma, rtol=0, atol=0
            ):
                raise AssertionError("600-episode checkpoint does not reproduce locked gamma=0.97")
            report["checkpoint_600_matches_locked_reference"] = True
    baseline = results / "baseline_evaluation.csv"
    if baseline.exists():
        frame = pd.read_csv(baseline)
        n_seeds = frame.model_seed.nunique()
        per_cell = frame.groupby(["policy", "gamma", "model_seed"]).size()
        if per_cell.nunique() != 1:
            raise AssertionError("baseline trace counts are not balanced")
        expected = (len(GAMMAS) + 1) * n_seeds * int(per_cell.iloc[0])
        if len(frame) != expected:
            raise AssertionError(f"baseline evaluation has {len(frame)} rows, expected {expected}")
        if set(frame.policy) != {"mpc4", "greedy8"}:
            raise AssertionError("baseline policy inventory mismatch")
    runtime = results / "runtime_benchmark.csv"
    if runtime.exists():
        frame = pd.read_csv(runtime)
        if set(frame.policy) != set(VARIANTS) | {"mpc4", "greedy8"}:
            raise AssertionError("runtime policy inventory mismatch")
        if set(frame.repeat) != {0, 1, 2} or not (frame.decision_ms >= 0).all():
            raise AssertionError("runtime repetition or duration mismatch")
        report["runtime_rows"] = len(frame)
        report["runtime_shared_samples"] = int(frame.sample_id.nunique())
    report["gamma_checkpoint_count"] = len(list((results / "gamma_models").glob("*.pth")))
    report["training_length_checkpoint_count"] = len(
        list((results / "training_length_models").glob("*.pth"))
    )
    report["status"] = "passed"
    (results / "revision_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    manifest_rows = []
    excluded = {"artifact_manifest.csv", "revision_validation.json"}
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.name not in excluded:
            manifest_rows.append({
                "file": path.relative_to(results).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    pd.DataFrame(manifest_rows).to_csv(results / "artifact_manifest.csv", index=False)
    print("revision_experiment_validation=passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("gamma", "evaluate", "length", "baselines", "runtime", "summary", "validate", "all"))
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seeds", default="800-819")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--traces", type=int, default=TRACE_COUNT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_seeds(spec: str) -> tuple[int, ...]:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return tuple(range(int(start), int(end) + 1))
    return tuple(int(x) for x in spec.split(",") if x)


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.results.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.stage in ("gamma", "all"):
        for gamma in GAMMAS:
            train_gamma(args.results, gamma, seeds, args.episodes, args.horizon, device, args.resume)
    if args.stage in ("evaluate", "all"):
        for gamma in GAMMAS:
            evaluate_gamma(args.results, gamma, seeds, args.episodes, args.horizon, args.traces, device)
    if args.stage in ("length", "all"):
        train_length(args.results, seeds, args.horizon, device, args.resume)
        evaluate_training_length(args.results, seeds, args.horizon, args.traces, device)
    if args.stage in ("baselines", "all"):
        evaluate_baselines(args.results, seeds, args.horizon, args.traces, args.resume)
    if args.stage in ("runtime", "all"):
        runtime_benchmark(args.results, seeds, args.horizon, device)
    if args.stage in ("summary", "all"):
        summarize(args.results)
    if args.stage in ("validate", "all"):
        validate(args.results)


if __name__ == "__main__":
    main()

