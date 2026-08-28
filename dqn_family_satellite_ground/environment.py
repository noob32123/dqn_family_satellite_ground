"""Sequential satellite-ground scheduling environment and pure previews."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 64
    coupling: float = 1.0
    scenario: str = "nominal"
    generator: str = "legacy_independent"
    compute_scale: float = 1.0
    data_scale: float = 1.0
    energy_scale: float = 1.0
    heat_scale: float = 1.0
    link_scale: float = 1.0


class SatelliteSchedulingEnv:
    task_dim = 15
    context_dim = 3
    resource_dim = 6
    state_dim = task_dim + context_dim + resource_dim
    action_dim = 3
    physics_task_names = (
        "onboard_heat_j",
        "workload_gflop",
        "onboard_compute_ms",
        "result_tx_ms_at_50mbps",
        "colocation_count_squared",
        "result_mbit",
        "ground_tx_heat_j",
        "ground_compute_ms",
        "raw_mbit",
        "preprocess_heat_j",
        "feature_tx_heat_j",
        "preprocess_work_gflop",
        "remaining_ground_ms",
        "total_workload_gflop",
        "feature_mbit",
    )
    physics_task_scales = np.array(
        [100, 4.8, 1400, 100, 2401, 4, 100, 100, 76.8,
         100, 100, 2.2, 100, 4.8, 32],
        dtype=np.float32,
    )

    def __init__(self, config: EnvConfig):
        self.config = config
        self.rng = np.random.default_rng()
        self.tasks = np.empty((0, self.task_dim), dtype=np.float32)
        self.link_trace = np.empty((0, 2), dtype=np.float32)
        self.task_primitives = np.empty((0, 9), dtype=np.float32)
        self.resources = np.zeros(self.resource_dim, dtype=np.float32)
        self.link_phase = 0.0
        self.t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.tasks = self._generate_tasks(self.config.horizon)
        self.link_trace = self._generate_link_trace(self.config.horizon + 1)
        initial_energy = 0.62 if self.config.scenario == "energy_limited" else 0.88
        initial_heat = 0.42 if self.config.scenario == "thermal_stress" else 0.22
        self.resources = np.array(
            [
                initial_heat,
                initial_energy,
                0.12,
                0.18,
                self.link_trace[0, 0],
                self.link_trace[0, 1],
            ],
            dtype=np.float32,
        )
        self.t = 0
        return self._state()

    def _generate_tasks(self, n: int) -> np.ndarray:
        if self.config.generator == "physics_correlated":
            return self._generate_physics_tasks(n)
        if self.config.generator != "legacy_independent":
            raise ValueError(f"unknown task generator: {self.config.generator}")
        r = self.rng
        x = np.empty((n, self.task_dim), dtype=np.float32)
        x[:, 0] = r.uniform(0, 1000, n)
        x[:, 1] = r.uniform(0, 1000, n)
        x[:, 2] = r.uniform(0, 1000, n)
        x[:, 3] = r.uniform(200, 300, n)
        x[:, 4] = r.integers(0, 50, n) ** 2
        x[:, 5] = r.uniform(0, 1000, n)
        x[:, 6] = r.uniform(0, 1000, n)
        x[:, 7] = r.uniform(200, 20000, n)
        x[:, 8] = r.uniform(0, 1000, n)
        total_heat = r.uniform(0, 1000, n)
        split = r.uniform(0.2, 0.8, n)
        x[:, 9] = total_heat * split
        x[:, 10] = total_heat * (1 - split)
        x[:, 11] = r.uniform(0, 1000, n)
        x[:, 12] = r.uniform(200, 20000, n)
        x[:, 13] = r.uniform(0, 1000, n)
        x[:, 14] = r.uniform(0, 1000, n)
        if self.config.scenario in {"burst", "thermal_stress"}:
            start, width = n // 3, max(4, n // 4)
            rows = slice(start, min(n, start + width))
            x[rows, [0, 1, 2, 9, 10, 11, 13]] *= 1.45
        return x

    def _generate_physics_tasks(self, n: int) -> np.ndarray:
        """Generate task descriptors from shared physical primitives.

        A Gaussian-copula-style latent pair couples input volume and arithmetic
        workload. All heat, energy, latency, and transmission descriptors are
        then deterministic functions of those primitives and of bounded
        compression/preprocessing ratios. This prevents independently sampled
        tuples that contradict the modeled work and data flows.
        """
        r = self.rng
        latent_data = r.normal(size=n)
        latent_work = 0.68 * latent_data + math.sqrt(1.0 - 0.68**2) * r.normal(size=n)
        data_factor = 1.0 / (1.0 + np.exp(-1.55 * latent_data))
        work_factor = 1.0 / (1.0 + np.exp(-1.55 * latent_work))

        raw_mbit = (8.0 + 56.0 * data_factor) * self.config.data_scale
        workload_gflop = (0.25 + 3.75 * work_factor) * self.config.compute_scale
        if self.config.scenario in {"burst", "thermal_stress"}:
            start, width = n // 3, max(4, n // 4)
            rows = slice(start, min(n, start + width))
            raw_mbit[rows] *= 1.45
            workload_gflop[rows] *= 1.45

        result_ratio = r.uniform(0.01, 0.05, n)
        feature_ratio = 0.08 + 0.22 * np.clip(
            0.65 * work_factor + 0.35 * r.uniform(size=n), 0.0, 1.0
        )
        preprocess_fraction = 0.20 + 0.25 * work_factor
        result_mbit = raw_mbit * result_ratio
        feature_mbit = raw_mbit * feature_ratio
        preprocess_work = workload_gflop * preprocess_fraction
        remaining_work = workload_gflop - preprocess_work

        onboard_rate_gflops = np.full(n, 4.0)
        ground_rate_gflops = np.full(n, 80.0)
        reference_link_mbps = np.full(n, 50.0)
        compute_power_w = 4.0 + 16.0 * np.power(
            np.clip(workload_gflop / 4.8, 0.0, 1.0), 0.70
        )
        transmit_power_w = np.full(n, 6.0)

        onboard_seconds = workload_gflop / onboard_rate_gflops
        ground_seconds = workload_gflop / ground_rate_gflops
        preprocess_seconds = preprocess_work / onboard_rate_gflops
        remaining_seconds = remaining_work / ground_rate_gflops
        result_tx_seconds = result_mbit / reference_link_mbps
        raw_tx_seconds = raw_mbit / reference_link_mbps
        feature_tx_seconds = feature_mbit / reference_link_mbps

        onboard_energy_j = compute_power_w * onboard_seconds
        ground_tx_energy_j = transmit_power_w * raw_tx_seconds
        preprocess_energy_j = compute_power_w * preprocess_seconds
        feature_tx_energy_j = transmit_power_w * feature_tx_seconds
        coloc_count = np.rint(6 + 43 * np.clip(
            0.60 * work_factor + 0.40 * data_factor, 0.0, 1.0
        )).astype(int)

        x = np.empty((n, self.task_dim), dtype=np.float32)
        x[:, 0] = 0.78 * onboard_energy_j
        x[:, 1] = workload_gflop
        x[:, 2] = 1000.0 * onboard_seconds
        x[:, 3] = 1000.0 * result_tx_seconds
        x[:, 4] = coloc_count**2
        x[:, 5] = result_mbit
        x[:, 6] = 0.55 * ground_tx_energy_j
        x[:, 7] = 1000.0 * ground_seconds
        x[:, 8] = raw_mbit
        x[:, 9] = 0.78 * preprocess_energy_j
        x[:, 10] = 0.55 * feature_tx_energy_j
        x[:, 11] = preprocess_work
        x[:, 12] = 1000.0 * remaining_seconds
        x[:, 13] = workload_gflop
        x[:, 14] = feature_mbit
        self.task_primitives = np.column_stack(
            (
                raw_mbit,
                workload_gflop,
                result_ratio,
                feature_ratio,
                preprocess_fraction,
                compute_power_w,
                onboard_rate_gflops,
                ground_rate_gflops,
                reference_link_mbps,
            )
        ).astype(np.float32)
        if not np.isfinite(x).all():
            raise FloatingPointError("non-finite physics-consistent task")
        return x

    def _generate_link_trace(self, n: int) -> np.ndarray:
        r = self.rng
        self.link_phase = float(r.uniform(0, 2 * np.pi))
        phase = self.link_phase
        index = np.arange(n)
        bandwidth = 0.62 + 0.23 * np.sin(2 * np.pi * index / 24 + phase)
        bandwidth += r.normal(0, 0.06, n)
        bandwidth *= self.config.link_scale
        if self.config.scenario == "link_limited":
            bandwidth -= 0.22
        bandwidth = np.clip(bandwidth, 0.12, 1.0)
        contact = 0.55 + 0.35 * np.sin(2 * np.pi * index / 31 + phase / 2)
        contact += r.normal(0, 0.04, n)
        contact = np.clip(contact, 0.08, 1.0)
        return np.column_stack((bandwidth, contact)).astype(np.float32)

    @staticmethod
    def _normalize_legacy_task(raw: np.ndarray) -> np.ndarray:
        scales = np.array(
            [
                1000, 1000, 1000, 300, 2401, 1000, 1000, 20000,
                1000, 1000, 1000, 1000, 20000, 1000, 1000,
            ],
            dtype=np.float32,
        )
        return np.clip(raw / scales, 0.0, 1.5).astype(np.float32)

    def _normalize_task(self, raw: np.ndarray) -> np.ndarray:
        if self.config.generator == "physics_correlated":
            return np.clip(raw / self.physics_task_scales, 0.0, 1.5).astype(np.float32)
        return self._normalize_legacy_task(raw)

    def _state(self) -> np.ndarray:
        task = self._normalize_task(self.tasks[min(self.t, len(self.tasks) - 1)])
        time_fraction = self.t / max(1, self.config.horizon - 1)
        context = np.array(
            [time_fraction, math.sin(self.link_phase), math.cos(self.link_phase)],
            dtype=np.float32,
        )
        return np.concatenate((task, context, self.resources)).astype(np.float32)

    def _immediate_costs_from(self, t: int, resources: np.ndarray) -> np.ndarray:
        x = self.tasks[t]
        heat, energy, queue, util, bandwidth, contact = resources
        eps = 0.08
        if self.config.generator == "physics_correlated":
            actual_rate = 2.2 + 217.8 * float(bandwidth)
            onboard_latency = x[2] + 1000.0 * x[5] / max(actual_rate, 2.2)
            ground_latency = x[7] + 1000.0 * x[8] / max(actual_rate, 2.2)
            hybrid_latency = (
                1000.0 * x[11] / 4.0 + x[12]
                + 1000.0 * x[14] / max(actual_rate, 2.2)
            )
            xn = np.clip(x / self.physics_task_scales, 0.0, 1.5)
            onboard = (
                0.70 * xn[0] + 0.65 * xn[1] + 0.00045 * onboard_latency
                + 0.55 * xn[4] + 0.35 * xn[5]
                + 0.35 * float(heat) + 0.12 * (1 - float(energy))
            )
            ground = (
                0.45 * xn[6] + 0.00045 * ground_latency + 0.55 * xn[8]
                + 0.35 * max(0.0, 0.28 - float(contact))
            )
            hybrid = (
                0.55 * (xn[9] + xn[10]) + 0.55 * xn[13]
                + 0.00045 * hybrid_latency + 0.45 * xn[14]
                + 0.18 * float(heat) + 0.12 * float(queue)
                + 0.18 * max(0.0, 0.22 - float(contact))
            )
            return np.array([onboard, ground, hybrid], dtype=np.float32)
        onboard = (
            0.4 * np.tan(np.pi / 2 * np.clip(x[0] / 5000, 0, 0.95))
            + 0.0004 * x[1]
            + 0.0002 * (x[2] * (1 + 1.4 * util + 0.7 * queue) + x[3])
            + 0.004 * x[4] * (1 + queue)
            + 0.4 * np.tan(np.pi / 2 * np.clip(x[5] / 5000, 0, 0.95))
            + 0.35 * heat
            + 0.12 * (1 - energy)
        )
        ground = (
            (2 / 3) * np.tan(np.pi / 2 * np.clip(x[6] / 5000, 0, 0.95))
            + 0.0005 * x[7] / max(float(bandwidth), eps)
            + 0.4 * np.tan(np.pi / 2 * np.clip(x[8] / 5000, 0, 0.95))
            + 0.35 * max(0.0, 0.28 - float(contact))
        )
        hybrid = (
            0.4 * np.tan(np.pi / 2 * np.clip((x[9] + x[10]) / 5000, 0, 0.95))
            + 0.0004 * x[13]
            + 0.0004 * x[11] * (1 + 0.8 * util)
            + 0.0004 * x[12] / max(float(bandwidth), eps)
            + 0.4 * np.tan(np.pi / 2 * np.clip(x[14] / 5000, 0, 0.95))
            + 0.18 * heat
            + 0.12 * queue
            + 0.18 * max(0.0, 0.22 - float(contact))
        )
        return np.array([onboard, ground, hybrid], dtype=np.float32)

    def immediate_costs(self) -> np.ndarray:
        return self._immediate_costs_from(self.t, self.resources)

    def _loads_from(self, t: int, resources: np.ndarray) -> tuple[np.ndarray, ...]:
        x = self.tasks[t]
        bandwidth = max(float(resources[4]), 0.1)
        if self.config.generator == "physics_correlated":
            work = np.array([x[1], 0.05 * x[1], x[13]]) / 4.8
            heat_load = np.array([x[0], 0.20 * x[6], x[9] + x[10]]) / 100.0
            data_load = np.array([x[5], x[8], x[14]]) / 76.8
            energy_use = self.config.energy_scale * np.array(
                [
                    0.012 + 0.042 * work[0],
                    0.007 + 0.025 * data_load[1] / bandwidth,
                    0.010 + 0.025 * work[2] + 0.018 * data_load[2] / bandwidth,
                ]
            )
            actual_rate = 2.2 + 217.8 * float(resources[4])
            latency_ms = np.array(
                [
                    x[2] + 1000.0 * x[5] / max(actual_rate, 2.2),
                    x[7] + 1000.0 * x[8] / max(actual_rate, 2.2),
                    1000.0 * x[11] / 4.0 + x[12]
                    + 1000.0 * x[14] / max(actual_rate, 2.2),
                ]
            )
            transmitted = np.array([x[5], x[8], x[14]])
            return work, self.config.heat_scale * heat_load, data_load, energy_use, latency_ms, transmitted
        work = np.array([x[1], 0.18 * x[7] / 20, x[13] + 0.25 * x[12] / 20]) / 1000
        heat_load = np.array([x[0], 0.08 * x[6], x[9] + x[10]]) / 1000
        data_load = np.array([0.20 * x[5], x[8], 0.65 * x[14]]) / 1000
        energy_use = self.config.energy_scale * np.array(
            [
                0.012 + 0.042 * work[0],
                0.007 + 0.025 * data_load[1] / bandwidth,
                0.010 + 0.025 * work[2] + 0.018 * data_load[2] / bandwidth,
            ]
        )
        latency_ms = np.array(
            [x[2] * (1 + 1.4 * resources[3] + 0.7 * resources[2]) + x[3],
             x[7] / bandwidth,
             x[11] * (1 + 0.8 * resources[3]) + x[12] / bandwidth]
        )
        transmitted = np.array([0.20 * x[5], x[8], 0.65 * x[14]]) / 1000
        return work, self.config.heat_scale * heat_load, data_load, energy_use, latency_ms, transmitted

    def transition_from(
        self, t: int, resources: np.ndarray, action: int
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Pure transition used by previews and finite-horizon planners."""
        if action not in (0, 1, 2):
            raise ValueError(f"invalid action {action}")
        if not 0 <= t < self.config.horizon:
            raise IndexError(f"transition time outside episode: {t}")
        immediate = float(self._immediate_costs_from(t, resources)[action])
        heat, energy, queue, util, bandwidth, contact = map(float, resources)
        coupling = max(0.0, float(self.config.coupling))
        work, heat_load, data_load, energy_use, latency_ms, transmitted = self._loads_from(t, resources)
        queue_add = np.array(
            [
                0.055 * work[0] * (1 + util),
                0.048 * data_load[1] / max(bandwidth, 0.1),
                0.030 * (work[2] + data_load[2] / max(bandwidth, 0.1)),
            ]
        )
        service = np.array(
            [
                0.030 * (1 - util),
                0.035 * bandwidth * contact,
                0.032 * (1 - 0.5 * util),
            ]
        )
        next_heat = np.clip(
            0.86 * heat + coupling * 0.17 * heat_load[action], 0, 1.3
        )
        next_energy = np.clip(
            energy + 0.006 - coupling * energy_use[action], 0, 1
        )
        next_queue = np.clip(
            queue + coupling * (queue_add[action] - service[action]), 0, 1.3
        )
        target_util = np.array(
            [work[0], 0.10 * data_load[1], 0.55 * work[2]]
        )[action]
        next_util = np.clip(0.68 * util + coupling * 0.32 * target_util, 0, 1.2)
        next_index = min(t + 1, self.config.horizon)
        base_bandwidth, base_contact = map(float, self.link_trace[next_index])
        occupation = np.array(
            [0.02, 0.18 * data_load[1], 0.12 * data_load[2]]
        )[action]
        next_bandwidth = np.clip(
            base_bandwidth - coupling * occupation, 0.08, 1
        )
        next_contact = np.clip(
            base_contact - coupling * 0.10 * data_load[action], 0.05, 1
        )
        thermal = max(0.0, next_heat - 0.78)
        depleted = max(0.0, 0.18 - next_energy)
        congested = max(0.0, next_queue - 0.75)
        unavailable = float(action != 0) * max(0.0, 0.14 - contact)
        penalty = coupling * (
            12.0 * thermal + 15.0 * depleted + 8.0 * congested + 10.0 * unavailable
        )
        total = immediate + penalty
        resources = np.array(
            [
                next_heat,
                next_energy,
                next_queue,
                next_util,
                next_bandwidth,
                next_contact,
            ],
            dtype=np.float32,
        )
        info = {
            "immediate_cost": immediate,
            "penalty": float(penalty),
            "total_cost": float(total),
            "thermal_violation": float(thermal > 0),
            "energy_violation": float(depleted > 0),
            "queue_violation": float(congested > 0),
            "contact_violation": float(unavailable > 0),
            "heat": float(next_heat),
            "energy": float(next_energy),
            "queue": float(next_queue),
            "energy_use": float(energy_use[action]),
            "latency_ms": float(latency_ms[action]),
            "transmitted_mbit": float(transmitted[action]),
        }
        return resources, info

    def _transition(self, action: int) -> tuple[np.ndarray, dict[str, float]]:
        return self.transition_from(self.t, self.resources, action)

    def preview_all_actions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current one-step outcomes without mutating state or RNG."""
        costs = np.empty(self.action_dim, dtype=np.float32)
        resources = np.empty((self.action_dim, self.resource_dim), dtype=np.float32)
        for action in range(self.action_dim):
            resources[action], info = self._transition(action)
            costs[action] = info["total_cost"]
        return costs, resources

    def snapshot(self) -> dict[str, Any]:
        """Testing helper that captures all mutable environment state."""
        return {
            "t": self.t,
            "resources": self.resources.copy(),
            "tasks": self.tasks.copy(),
            "link_trace": self.link_trace.copy(),
            "link_phase": self.link_phase,
            "task_primitives": self.task_primitives.copy(),
            "rng": self.rng.bit_generator.state,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        next_resources, info = self._transition(action)
        self.resources = next_resources
        self.t += 1
        done = self.t >= self.config.horizon
        next_state = (
            np.zeros(self.state_dim, dtype=np.float32) if done else self._state()
        )
        return next_state, -info["total_cost"], done, info
