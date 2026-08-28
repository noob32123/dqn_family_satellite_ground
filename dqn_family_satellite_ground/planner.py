"""Deterministic finite-horizon planning baselines for reviewer controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import SatelliteSchedulingEnv


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 4
    gamma: float = 0.97


class RecedingHorizonPlanner:
    """Exact enumeration over a short perfect-forecast receding horizon.

    The planner is deliberately separate from the learning algorithms. At each
    real decision it enumerates all 3^H action sequences over the next H known
    tasks and link states, applies only the first action, and replans. H=4 is
    exact for the stated four-step objective and provides a strong model-based
    control comparator without claiming global optimality over 64 steps.
    """

    def __init__(self, config: PlannerConfig = PlannerConfig()):
        if config.horizon < 1:
            raise ValueError("planner horizon must be positive")
        if not 0.0 <= config.gamma <= 1.0:
            raise ValueError("planner gamma must lie in [0, 1]")
        self.config = config

    def act(self, env: SatelliteSchedulingEnv) -> int:
        remaining = env.config.horizon - env.t
        depth = min(self.config.horizon, remaining)
        frontier: list[tuple[float, np.ndarray, int]] = [
            (0.0, env.resources.copy(), -1)
        ]
        for offset in range(depth):
            expanded: list[tuple[float, np.ndarray, int]] = []
            for cumulative, resources, first_action in frontier:
                for action in range(env.action_dim):
                    next_resources, info = env.transition_from(
                        env.t + offset, resources, action
                    )
                    initial = action if first_action < 0 else first_action
                    expanded.append(
                        (
                            cumulative
                            + self.config.gamma**offset * info["total_cost"],
                            next_resources,
                            initial,
                        )
                    )
            frontier = expanded
        best = min(frontier, key=lambda item: (item[0], item[2]))
        return int(best[2])

