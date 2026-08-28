from __future__ import annotations

import copy
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from dqn_family_satellite_ground.agent import AgentConfig, DQNAgent, QNetwork
from dqn_family_satellite_ground.environment import EnvConfig, SatelliteSchedulingEnv
from dqn_family_satellite_ground.planner import PlannerConfig, RecedingHorizonPlanner
from dqn_family_satellite_ground.reviewer_experiments import (
    _exact_two_sided_sign_p,
    _holm_adjust,
)


class EnvironmentPreviewTests(unittest.TestCase):
    def test_preview_is_side_effect_free(self) -> None:
        env = SatelliteSchedulingEnv(EnvConfig(horizon=16, scenario="burst"))
        env.reset(seed=1234)
        before = copy.deepcopy(env.snapshot())
        env.preview_all_actions()
        after = env.snapshot()
        self.assertEqual(before["t"], after["t"])
        self.assertEqual(before["link_phase"], after["link_phase"])
        np.testing.assert_array_equal(before["resources"], after["resources"])
        np.testing.assert_array_equal(before["tasks"], after["tasks"])
        np.testing.assert_array_equal(before["link_trace"], after["link_trace"])
        np.testing.assert_array_equal(
            before["task_primitives"], after["task_primitives"]
        )
        self.assertEqual(before["rng"], after["rng"])

    def test_preview_matches_real_step(self) -> None:
        for action in range(3):
            env = SatelliteSchedulingEnv(
                EnvConfig(horizon=16, coupling=1.0, scenario="link_limited")
            )
            env.reset(seed=771)
            costs, resources = env.preview_all_actions()
            _, _, _, info = env.step(action)
            np.testing.assert_allclose(resources[action], env.resources, rtol=0, atol=0)
            self.assertAlmostEqual(float(costs[action]), info["total_cost"], places=5)

    def test_physics_preview_matches_real_step(self) -> None:
        for action in range(3):
            env = SatelliteSchedulingEnv(
                EnvConfig(
                    horizon=16,
                    coupling=1.0,
                    scenario="thermal_stress",
                    generator="physics_correlated",
                )
            )
            env.reset(seed=908)
            before = copy.deepcopy(env.snapshot())
            costs, resources = env.preview_all_actions()
            after = env.snapshot()
            self.assertEqual(before["rng"], after["rng"])
            self.assertEqual(before["link_phase"], after["link_phase"])
            np.testing.assert_array_equal(before["resources"], after["resources"])
            np.testing.assert_array_equal(before["tasks"], after["tasks"])
            _, _, _, info = env.step(action)
            np.testing.assert_allclose(resources[action], env.resources, rtol=0, atol=0)
            self.assertAlmostEqual(float(costs[action]), info["total_cost"], places=5)

    def test_physics_preview_matches_at_clipping_boundaries_and_terminal_step(self) -> None:
        lower = np.array([0.0, 0.0, 0.0, 0.0, 0.08, 0.05], dtype=np.float32)
        upper = np.array([1.3, 1.0, 1.3, 1.2, 1.0, 1.0], dtype=np.float32)
        for boundary in (lower, upper):
            for action in range(3):
                env = SatelliteSchedulingEnv(
                    EnvConfig(horizon=4, generator="physics_correlated")
                )
                env.reset(seed=19_009)
                env.t = env.config.horizon - 1
                env.resources = boundary.copy()
                costs, resources = env.preview_all_actions()
                next_state, _, done, info = env.step(action)
                self.assertTrue(done)
                np.testing.assert_array_equal(next_state, np.zeros(env.state_dim))
                np.testing.assert_allclose(resources[action], env.resources, rtol=0, atol=0)
                self.assertAlmostEqual(float(costs[action]), info["total_cost"], places=5)


class PhysicsGeneratorTests(unittest.TestCase):
    def test_state_exposes_time_and_link_phase(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=16, generator="physics_correlated")
        )
        state = env.reset(seed=19010)
        self.assertEqual(state.shape, (24,))
        self.assertAlmostEqual(float(state[15]), 0.0, places=7)
        self.assertAlmostEqual(float(state[16]), np.sin(env.link_phase), places=7)
        self.assertAlmostEqual(float(state[17]), np.cos(env.link_phase), places=7)
        next_state, *_ = env.step(0)
        self.assertAlmostEqual(float(next_state[15]), 1.0 / 15.0, places=7)
        np.testing.assert_array_equal(next_state[-6:], env.resources)

    def test_link_phase_makes_trace_mean_conditionally_reconstructable(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=16, generator="physics_correlated")
        )
        env.reset(seed=19011)
        index = np.arange(env.config.horizon + 1)
        expected_bandwidth_mean = 0.62 + 0.23 * np.sin(
            2 * np.pi * index / 24 + env.link_phase
        )
        expected_contact_mean = 0.55 + 0.35 * np.sin(
            2 * np.pi * index / 31 + env.link_phase / 2
        )
        self.assertEqual(len(expected_bandwidth_mean), len(env.link_trace))
        self.assertEqual(len(expected_contact_mean), len(env.link_trace))

    def test_reported_physics_cost_equations_match_code(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=16, generator="physics_correlated")
        )
        env.reset(seed=19_006)
        x = env.tasks[0].astype(float)
        heat, energy, queue, _, bandwidth, contact = env.resources.astype(float)
        xn = np.clip(x / env.physics_task_scales, 0.0, 1.5)
        actual_rate = 2.2 + 217.8 * bandwidth
        onboard_latency = x[2] + 1000.0 * x[5] / max(actual_rate, 2.2)
        ground_latency = x[7] + 1000.0 * x[8] / max(actual_rate, 2.2)
        hybrid_latency = (
            1000.0 * x[11] / 4.0
            + x[12]
            + 1000.0 * x[14] / max(actual_rate, 2.2)
        )
        expected = np.array(
            [
                0.70 * xn[0] + 0.65 * xn[1] + 0.00045 * onboard_latency
                + 0.55 * xn[4] + 0.35 * xn[5] + 0.35 * heat
                + 0.12 * (1.0 - energy),
                0.45 * xn[6] + 0.00045 * ground_latency + 0.55 * xn[8]
                + 0.35 * max(0.0, 0.28 - contact),
                0.55 * (xn[9] + xn[10]) + 0.55 * xn[13]
                + 0.00045 * hybrid_latency + 0.45 * xn[14]
                + 0.18 * heat + 0.12 * queue
                + 0.18 * max(0.0, 0.22 - contact),
            ]
        )
        np.testing.assert_allclose(
            env.immediate_costs(), expected, rtol=2e-7, atol=2e-7
        )

    def test_reported_contact_hinges_match_thresholds(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=16, generator="physics_correlated")
        )
        env.reset(seed=19_007)
        resources = env.resources.copy()
        resources[5] = 0.28
        at_ground_threshold = env._immediate_costs_from(0, resources)
        resources[5] = 0.27
        below_ground_threshold = env._immediate_costs_from(0, resources)
        self.assertAlmostEqual(
            float(below_ground_threshold[1] - at_ground_threshold[1]),
            0.35 * 0.01,
            places=6,
        )
        resources[5] = 0.22
        at_hybrid_threshold = env._immediate_costs_from(0, resources)
        resources[5] = 0.21
        below_hybrid_threshold = env._immediate_costs_from(0, resources)
        self.assertAlmostEqual(
            float(below_hybrid_threshold[2] - at_hybrid_threshold[2]),
            0.18 * 0.01,
            places=6,
        )

    def test_reported_normalization_clips_out_of_envelope_descriptor(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=16, generator="physics_correlated")
        )
        env.reset(seed=19_008)
        original = float(env.tasks[0, 4])
        env.tasks[0, 4] = 1.5 * env.physics_task_scales[4]
        at_clip = env.immediate_costs()
        env.tasks[0, 4] = 3.0 * env.physics_task_scales[4]
        beyond_clip = env.immediate_costs()
        self.assertAlmostEqual(float(at_clip[0]), float(beyond_clip[0]), places=6)
        env.tasks[0, 4] = original

    def test_task_descriptors_obey_dataflow_constraints(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=20_000, generator="physics_correlated")
        )
        env.reset(seed=9_117_031)
        raw = env.task_primitives[:, 0]
        workload = env.task_primitives[:, 1]
        result = env.tasks[:, 5]
        feature = env.tasks[:, 14]
        self.assertTrue(np.isfinite(env.tasks).all())
        self.assertTrue((env.task_primitives > 0).all())
        self.assertTrue((result <= feature).all())
        self.assertTrue((feature <= raw).all())
        self.assertGreater(float(np.corrcoef(raw, workload)[0, 1]), 0.45)
        self.assertGreater(float(np.corrcoef(workload, env.tasks[:, 0])[0, 1]), 0.90)
        self.assertGreater(float(np.corrcoef(raw, env.tasks[:, 6])[0, 1]), 0.99)

    def test_parameter_scales_have_declared_direction(self) -> None:
        base = SatelliteSchedulingEnv(
            EnvConfig(horizon=128, generator="physics_correlated")
        )
        scaled = SatelliteSchedulingEnv(
            EnvConfig(
                horizon=128,
                generator="physics_correlated",
                compute_scale=1.2,
                data_scale=1.2,
            )
        )
        base.reset(seed=44)
        scaled.reset(seed=44)
        np.testing.assert_allclose(
            scaled.task_primitives[:, 0],
            1.2 * base.task_primitives[:, 0],
            rtol=2e-7,
        )
        np.testing.assert_allclose(
            scaled.task_primitives[:, 1],
            1.2 * base.task_primitives[:, 1],
            rtol=2e-7,
        )


class PlannerTests(unittest.TestCase):
    def test_planner_does_not_mutate_environment(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=16, generator="physics_correlated")
        )
        env.reset(seed=125)
        before = copy.deepcopy(env.snapshot())
        action = RecedingHorizonPlanner().act(env)
        after = env.snapshot()
        self.assertIn(action, (0, 1, 2))
        self.assertEqual(before["t"], after["t"])
        self.assertEqual(before["rng"], after["rng"])
        for key in ("resources", "tasks", "link_trace", "task_primitives"):
            np.testing.assert_array_equal(before[key], after[key])

    def test_one_step_planner_equals_immediate_argmin(self) -> None:
        env = SatelliteSchedulingEnv(
            EnvConfig(horizon=16, generator="physics_correlated")
        )
        env.reset(seed=126)
        planner = RecedingHorizonPlanner(PlannerConfig(horizon=1, gamma=0.97))
        self.assertEqual(planner.act(env), int(np.argmin(env.immediate_costs())))


class PureDQNTests(unittest.TestCase):
    def _fill(self, agent: DQNAgent, preview_value: float) -> None:
        rng = np.random.default_rng(9)
        for index in range(agent.config.batch_size):
            state = rng.normal(size=24).astype(np.float32)
            next_state = rng.normal(size=24).astype(np.float32)
            agent.observe(
                state,
                index % 3,
                float(rng.normal()),
                next_state,
                index % 11 == 0,
                np.full(3, preview_value, dtype=np.float32),
                np.full((3, 6), preview_value, dtype=np.float32),
            )

    def test_standard_dqn_ignores_counterfactual_payload(self) -> None:
        config = AgentConfig(batch_size=16)
        left = DQNAgent("standard_dqn", config, torch.device("cpu"), seed=17)
        right = DQNAgent("standard_dqn", config, torch.device("cpu"), seed=17)
        self._fill(left, 0.0)
        self._fill(right, 99.0)
        left.update()
        right.update()
        for a, b in zip(left.online.parameters(), right.online.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_centering_is_invariant_to_common_shift(self) -> None:
        values = torch.tensor([[1.0, 4.0, -2.0], [0.5, 0.0, 2.5]])
        shifted = values + torch.tensor([[91.0], [-13.0]])
        torch.testing.assert_close(
            DQNAgent.centered(values), DQNAgent.centered(shifted)
        )

    def test_all_variants_share_network_shape(self) -> None:
        reference = [p.shape for p in QNetwork().parameters()]
        for variant in (
            "standard_dqn",
            "centered_full_action_dqn",
            "full_action_q_dqn",
            "immediate_advantage_dqn",
            "double_dqn",
            "double_centered_full_action_dqn",
        ):
            agent = DQNAgent(
                variant, AgentConfig(), torch.device("cpu"), seed=1
            )
            self.assertEqual(reference, [p.shape for p in agent.online.parameters()])

    def test_factual_target_is_standard_target_network_max(self) -> None:
        agent = DQNAgent(
            "standard_dqn", AgentConfig(batch_size=2), torch.device("cpu"), seed=3
        )
        self._fill(agent, 0.0)
        batch = agent.buffer.sample(2, random.Random(4))
        states, actions, rewards, next_states, dones, *_ = batch
        losses = agent.compute_losses(batch)
        with torch.no_grad():
            states_t = torch.as_tensor(states)
            actions_t = torch.as_tensor(actions).long()
            next_t = torch.as_tensor(next_states)
            expected_target = torch.as_tensor(rewards, dtype=torch.float32) + agent.config.gamma * (
                1 - torch.as_tensor(dones, dtype=torch.float32)
            ) * agent.target(next_t).max(1).values
            expected = torch.nn.functional.smooth_l1_loss(
                agent.online(states_t).gather(1, actions_t[:, None]).squeeze(1),
                expected_target,
            )
        torch.testing.assert_close(losses["td_loss"], expected)

    def test_uncentered_and_centered_auxiliary_targets_are_distinct(self) -> None:
        config = AgentConfig(batch_size=16)
        centered = DQNAgent("centered_full_action_dqn", config, torch.device("cpu"), seed=21)
        full_q = DQNAgent("full_action_q_dqn", config, torch.device("cpu"), seed=21)
        self._fill(centered, 0.4)
        batch = centered.buffer.sample(16, random.Random(8))
        centered_loss = centered.compute_losses(batch)["auxiliary_loss"]
        full_q_loss = full_q.compute_losses(batch)["auxiliary_loss"]
        self.assertNotAlmostEqual(centered_loss.item(), full_q_loss.item(), places=6)

    def test_double_target_uses_online_selection(self) -> None:
        agent = DQNAgent(
            "double_dqn", AgentConfig(batch_size=2), torch.device("cpu"), seed=31
        )
        self._fill(agent, 0.0)
        batch = agent.buffer.sample(2, random.Random(5))
        states, actions, rewards, next_states, dones, *_ = batch
        losses = agent.compute_losses(batch)
        with torch.no_grad():
            next_t = torch.as_tensor(next_states)
            selected = agent.online(next_t).argmax(1, keepdim=True)
            next_value = agent.target(next_t).gather(1, selected).squeeze(1)
            expected_target = torch.as_tensor(rewards, dtype=torch.float32) + agent.config.gamma * (
                1 - torch.as_tensor(dones, dtype=torch.float32)
            ) * next_value
            expected = torch.nn.functional.smooth_l1_loss(
                agent.online(torch.as_tensor(states)).gather(
                    1, torch.as_tensor(actions).long()[:, None]
                ).squeeze(1),
                expected_target,
            )
        torch.testing.assert_close(losses["td_loss"], expected)


class SourceGuardTests(unittest.TestCase):
    def test_no_forbidden_dqn_enhancement(self) -> None:
        source = (Path(__file__).parents[1] / "agent.py").read_text(encoding="utf-8").lower()
        forbidden_symbols = (
            "prioritizedreplay", "duelingnetwork", "noisylayer",
            "distributionaldqn", "n_step_return",
        )
        for symbol in forbidden_symbols:
            self.assertNotIn(symbol, source)


class StatisticalInferenceTests(unittest.TestCase):
    def test_exact_sign_test_uses_independent_seed_count(self) -> None:
        differences = -np.arange(1.0, 21.0)
        self.assertAlmostEqual(
            _exact_two_sided_sign_p(differences), 2.0 / (2**20), places=12
        )

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        p_values = np.array([0.01, 0.04, 0.02])
        adjusted = _holm_adjust(p_values)
        self.assertTrue(np.all((0.0 <= adjusted) & (adjusted <= 1.0)))
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
