"""Write fixed-seed one-step transition examples for all scheduling actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .environment import EnvConfig, SatelliteSchedulingEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "reviewer_revision_state_complete"
        / "fixed_transition_audit.csv",
    )
    args = parser.parse_args()
    env = SatelliteSchedulingEnv(
        EnvConfig(horizon=64, coupling=1.0, generator="physics_correlated")
    )
    state = env.reset(seed=19_010)
    costs, preview_resources = env.preview_all_actions()
    names = ("onboard", "ground", "collaborative")
    rows: list[dict] = []
    for action, name in enumerate(names):
        replay = SatelliteSchedulingEnv(env.config)
        replay.reset(seed=19_010)
        next_state, reward, done, info = replay.step(action)
        row = {
            "seed": 19_010,
            "time_index": 0,
            "action": action,
            "action_name": name,
            "state_task_1": float(state[0]),
            "state_heat": float(state[-6]),
            "state_energy": float(state[-5]),
            "preview_total_cost": float(costs[action]),
            "executed_total_cost": float(info["total_cost"]),
            "reward": float(reward),
            "done": bool(done),
        }
        for index, resource in enumerate(
            ("heat", "energy", "queue", "utilization", "bandwidth", "contact")
        ):
            row[f"preview_{resource}"] = float(preview_resources[action, index])
            row[f"executed_{resource}"] = float(next_state[-6 + index])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
