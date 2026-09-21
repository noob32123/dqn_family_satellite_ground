"""Behavioral checks for the new protocols, independent of measured results."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dqn_family_satellite_ground.experiment import config_for, save_checkpoint, train_model
from dqn_family_satellite_ground.planner import PlannerConfig, RecedingHorizonPlanner
from dqn_family_satellite_ground.reviewer_experiments import revision_env
from dqn_family_satellite_ground.environment import SatelliteSchedulingEnv
from dqn_family_satellite_ground.revision_experiments import greedy_rollout_action, trace_seed_for
from dqn_family_satellite_ground.revision_experiments import paired_inference


class RevisionProtocolTests(unittest.TestCase):
    def test_trace_namespace_matches_locked_evaluation(self):
        self.assertEqual(trace_seed_for(800, 0), 3_300_000)
        self.assertEqual(trace_seed_for(819, 19), 3_319_019)
        self.assertEqual(len({trace_seed_for(s, j) for s in range(800, 820)
                              for j in range(20)}), 400)

    def test_fixed_schedule_and_original_defaults(self):
        self.assertEqual(config_for("standard_dqn").epsilon_decay_steps, 13440)
        config = config_for("double_dqn", 900, 64, gamma_override=.95,
                            epsilon_decay_steps_override=13440)
        self.assertEqual(config.gamma, .95)
        self.assertEqual(config.epsilon_decay_steps, 13440)

    def test_rollout_does_not_modify_environment_or_random_stream(self):
        env = SatelliteSchedulingEnv(revision_env(12))
        env.reset(seed=991)
        before = copy.deepcopy(env.snapshot())
        action = greedy_rollout_action(env)
        after = env.snapshot()
        self.assertIn(action, (0, 1, 2))
        for key in before:
            if isinstance(before[key], np.ndarray):
                np.testing.assert_array_equal(before[key], after[key])
            else:
                self.assertEqual(before[key], after[key])

    def test_rollout_equals_exhaustive_planner_at_last_two_steps(self):
        # With only one future action, greedy completion enumerates every
        # possible two-action sequence and must match exact MPC.
        for seed in range(10):
            env = SatelliteSchedulingEnv(revision_env(8))
            env.reset(seed=seed)
            env.t = 6
            self.assertEqual(greedy_rollout_action(env),
                             RecedingHorizonPlanner(PlannerConfig(2, .97)).act(env))
            env.t = 7
            self.assertEqual(greedy_rollout_action(env),
                             RecedingHorizonPlanner(PlannerConfig(1, .97)).act(env))

    def test_checkpoint_writes_do_not_change_training_trajectory(self):
        cpu = torch.device("cpu")
        torch.set_num_threads(1)
        ordinary, curve = train_model(
            "standard_dqn", 991, 5, 32, cpu,
            environment_config=revision_env(32),
            epsilon_decay_steps_override=13440,
        )
        with tempfile.TemporaryDirectory() as tmp:
            def callback(agent, rows, episode):
                save_checkpoint(Path(tmp) / f"{episode}.pth", agent, 991, 5, 32)
                pd.DataFrame(rows).to_csv(Path(tmp) / f"{episode}.csv", index=False)

            checkpointed, callback_curve = train_model(
                "standard_dqn", 991, 5, 32, cpu,
                environment_config=revision_env(32),
                epsilon_decay_steps_override=13440, episode_callback=callback,
            )
            self.assertEqual(len(list(Path(tmp).glob("*.pth"))), 5)
        pd.testing.assert_frame_equal(pd.DataFrame(curve), pd.DataFrame(callback_curve))
        for key, value in ordinary.online.state_dict().items():
            torch.testing.assert_close(value, checkpointed.online.state_dict()[key],
                                       rtol=0, atol=0)
        self.assertEqual(ordinary.rng.getstate(), checkpointed.rng.getstate())

    def test_added_inference_uses_seed_means_as_pairs(self):
        rows = []
        for variant in ("a", "b"):
            for gamma, shift in ((.97, 0.0), (.90, -1.0)):
                for seed in range(20):
                    rows.append({"variant": variant, "gamma": gamma,
                                 "model_seed": seed,
                                 "total_cost": 10.0 + seed / 10 + shift})
        with tempfile.TemporaryDirectory() as tmp:
            result = paired_inference(
                pd.DataFrame(rows), ["variant"], "gamma", .97,
                Path(tmp) / "inference.csv",
            )
        self.assertEqual(len(result), 2)
        self.assertTrue((result.n_pairs == 20).all())
        np.testing.assert_allclose(result.mean_difference, -1.0)
        self.assertTrue((result.ci95_high < 0).all())


if __name__ == "__main__":
    unittest.main()
