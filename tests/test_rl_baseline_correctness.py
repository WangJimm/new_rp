from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


# The desktop verification environment does not bundle PyTorch.  The PPO
# rollout buffer itself only needs numpy, so provide the minimal import shim
# required to load that module without changing production behavior.
try:
    import torch  # type: ignore  # noqa: F401
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    fake_torch = types.ModuleType("torch")

    def no_grad():
        def decorate(function):
            return function
        return decorate

    fake_torch.no_grad = no_grad
    sys.modules["torch"] = fake_torch


REFACTOR_ROOT = Path(__file__).resolve().parents[1]
if str(REFACTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(REFACTOR_ROOT))

from DQN.ppo_common import compute_gae
from DQN.ppo_rllib import PPORTrajectoryMemory, RLLibPPOAgentRefactored

if TORCH_AVAILABLE:
    from DQN.ppo_torch import TorchPPOAgentRefactored, TorchPPOShapeConfig


class PPORTrajectoryMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = PPORTrajectoryMemory(action_dim=3, capacity=100)

    @staticmethod
    def payload(value: float, *, done: bool = False):
        state = np.asarray([value, 0.0], dtype=np.float32)
        next_state = np.asarray([value + 1.0, 0.0], dtype=np.float32)
        return (
            state,
            np.asarray([0], dtype=np.int64),
            next_state,
            np.asarray([value], dtype=np.float32),
            np.asarray([float(done)], dtype=np.float32),
            np.asarray([True, True, False], dtype=np.bool_),
            None,
            None,
            (0, 1, [10, 11], None),
            None,
        )

    def push(self, key: str, timestep: int, value: float, *, terminal: bool = False):
        self.memory.push_trajectory(
            *self.payload(value, done=terminal),
            trajectory_key=key,
            timestep=timestep,
            terminated=terminal,
        )

    def test_interleaved_evs_are_drained_as_independent_ordered_trajectories(self):
        self.push("rollout:EV_A", 0, 1.0)
        self.push("rollout:EV_B", 0, 10.0)
        self.push("rollout:EV_A", 1, 2.0, terminal=True)
        self.push("rollout:EV_B", 1, 20.0, terminal=True)

        self.assertEqual(self.memory.complete_size, 4)
        trajectories = self.memory.drain_complete()

        self.assertEqual(len(trajectories), 2)
        self.assertEqual([[int(item["timestep"]) for item in traj] for traj in trajectories], [[0, 1], [0, 1]])
        self.assertNotEqual(trajectories[0][0]["trajectory_id"], trajectories[1][0]["trajectory_id"])
        self.assertTrue(all(bool(traj[-1]["terminated"]) for traj in trajectories))
        self.assertEqual(len(self.memory), 0)

    def test_incomplete_trajectory_is_retained_when_complete_one_is_drained(self):
        self.push("rollout:EV_A", 0, 1.0)
        self.push("rollout:EV_B", 0, 10.0, terminal=True)

        trajectories = self.memory.drain_complete()

        self.assertEqual(len(trajectories), 1)
        self.assertEqual(float(trajectories[0][0]["reward"]), 10.0)
        self.assertEqual(len(self.memory), 1)
        self.assertEqual(self.memory.complete_size, 0)

    def test_checkpoint_roundtrip_preserves_trajectory_identity(self):
        self.push("rollout:EV_A", 0, 1.0)
        restored = PPORTrajectoryMemory(action_dim=1)
        restored.load_state_dict(self.memory.state_dict())
        restored.push_trajectory(
            *self.payload(2.0, done=True),
            trajectory_key="rollout:EV_A",
            timestep=1,
            terminated=True,
        )

        trajectory = restored.drain_complete()[0]
        self.assertEqual([int(item["timestep"]) for item in trajectory], [0, 1])
        self.assertEqual(len({int(item["trajectory_id"]) for item in trajectory}), 1)

    def test_legacy_checkpoint_without_trajectory_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "start a fresh PPO run"):
            self.memory.load_state_dict({"items": [{"state": np.zeros(1)}]})

    def test_numpy_neighbor_array_builds_mask_without_ambiguous_truth_value(self):
        payload = list(self.payload(1.0, done=True))
        payload[8] = (0, 1, np.asarray([10, 11]), None)
        self.memory.push_trajectory(
            *payload,
            trajectory_key="rollout:EV_A",
            timestep=0,
            terminated=True,
        )
        np.testing.assert_array_equal(
            self.memory.items[0]["action_mask"],
            np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        )


class PPOMathTests(unittest.TestCase):
    def test_terminal_gae_does_not_bootstrap_past_terminal(self):
        advantages, returns = compute_gae(
            np.asarray([1.0, 1.0], dtype=np.float32),
            np.asarray([0.5, 0.25], dtype=np.float32),
            np.asarray([False, True]),
            bootstrap_value=99.0,
            gamma=1.0,
            gae_lambda=1.0,
        )
        np.testing.assert_allclose(advantages, [1.5, 0.75], atol=1e-6)
        np.testing.assert_allclose(returns, [2.0, 1.0], atol=1e-6)

    def test_truncated_gae_bootstraps_final_next_value(self):
        advantages, returns = compute_gae(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
            np.asarray([False]),
            bootstrap_value=0.4,
            gamma=0.9,
            gae_lambda=0.95,
        )
        np.testing.assert_allclose(advantages, [0.86], atol=1e-6)
        np.testing.assert_allclose(returns, [1.36], atol=1e-6)


class RLLibPPOUpdateTests(unittest.TestCase):
    def build_agent(self, policy):
        memory = PPORTrajectoryMemory(action_dim=3, capacity=100)
        for key, value in (("rollout:EV_A", 1.0), ("rollout:EV_B", 2.0)):
            state = np.asarray([value, 0.0], dtype=np.float32)
            memory.push_trajectory(
                state,
                np.asarray([0], dtype=np.int64),
                state + 1.0,
                np.asarray([value], dtype=np.float32),
                np.asarray([1.0], dtype=np.float32),
                np.asarray([True, True, False], dtype=np.bool_),
                None,
                None,
                (0, 1, [10, 11], None),
                None,
                trajectory_key=key,
                timestep=0,
                terminated=True,
            )

        class Postprocessing:
            ADVANTAGES = "advantages"

        calls = {"concat": 0, "standardized": 0}

        def concat_samples(samples):
            calls["concat"] += 1
            return {
                "advantages": np.asarray(
                    [float(sample["trajectory"][0]["reward"]) for sample in samples],
                    dtype=np.float32,
                )
            }

        def standardized(values):
            calls["standardized"] += 1
            return values - values.mean()

        def minibatches(batch, size, shuffle=True):
            yield batch

        agent = object.__new__(RLLibPPOAgentRefactored)
        agent.memory = memory
        agent.train_batch_size = 2
        agent.sgd_minibatch_size = 2
        agent.num_sgd_iter = 2
        agent.update_attempt_counter = 0
        agent.learn_step_counter = 0
        agent.gradient_step_counter = 0
        agent._ensure_algorithm = lambda: None
        agent._policy = lambda: policy
        agent._build_sample_batch = lambda trajectory: {"trajectory": trajectory}
        agent._rllib_modules = lambda: {
            "Postprocessing": Postprocessing,
            "concat_samples": concat_samples,
            "standardized": standardized,
            "minibatches": minibatches,
        }
        return agent, calls

    def test_failed_update_retains_complete_on_policy_trajectories(self):
        class FailingPolicy:
            def postprocess_trajectory(self, batch):
                return batch

            def learn_on_batch(self, batch):
                raise AttributeError("simulated RLlib failure")

        agent, calls = self.build_agent(FailingPolicy())
        with self.assertRaisesRegex(AttributeError, "simulated RLlib failure"):
            agent.learn_dqn_neighbor()

        self.assertEqual(len(agent.memory), 2)
        self.assertEqual(agent.memory.complete_size, 2)
        self.assertEqual(agent.update_attempt_counter, 1)
        self.assertEqual(agent.learn_step_counter, 0)
        self.assertEqual(agent.gradient_step_counter, 0)
        self.assertEqual(calls["concat"], 1)

    def test_successful_update_uses_module_concat_runs_epochs_and_commits_buffer(self):
        class SuccessfulPolicy:
            def postprocess_trajectory(self, batch):
                return batch

            def learn_on_batch(self, batch):
                return {"learner_stats": {"total_loss": np.float32(2.5)}}

        agent, calls = self.build_agent(SuccessfulPolicy())
        loss = agent.learn_dqn_neighbor()

        self.assertAlmostEqual(loss, 2.5)
        self.assertEqual(len(agent.memory), 0)
        self.assertEqual(agent.update_attempt_counter, 1)
        self.assertEqual(agent.learn_step_counter, 1)
        self.assertEqual(agent.gradient_step_counter, 2)
        self.assertEqual(calls["concat"], 1)
        self.assertEqual(calls["standardized"], 1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the desktop test runtime")
class TorchPPOUpdateTests(unittest.TestCase):
    def test_small_rollout_performs_clipped_ppo_update_without_ray(self):
        shape = TorchPPOShapeConfig(num_nodes=2, max_neighbors=2)
        agent = TorchPPOAgentRefactored(
            shape,
            train_batch_size=4,
            sgd_minibatch_size=2,
            num_sgd_iter=2,
            device=torch.device("cpu"),
        )
        for trajectory_index in range(2):
            key = f"rollout:EV_{trajectory_index}"
            for timestep in range(2):
                terminal = timestep == 1
                state = np.asarray(
                    [1.0, 0.0, 0.0, 1.0, 0.1 * timestep],
                    dtype=np.float32,
                )
                agent.memory.push_trajectory(
                    state,
                    np.asarray([timestep % 2], dtype=np.int64),
                    state.copy(),
                    np.asarray([1.0 if terminal else -0.1], dtype=np.float32),
                    np.asarray([float(terminal)], dtype=np.float32),
                    np.asarray([True, True], dtype=np.bool_),
                    None,
                    None,
                    (0, 1, [0, 1], None),
                    None,
                    trajectory_key=key,
                    timestep=timestep,
                    terminated=terminal,
                )

        loss = agent.learn_dqn_neighbor()

        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(agent.memory), 0)
        self.assertEqual(agent.learn_step_counter, 1)
        self.assertEqual(agent.gradient_step_counter, 4)
        self.assertEqual(agent.last_update_metrics["batch_size"], 4.0)


if __name__ == "__main__":
    unittest.main()
