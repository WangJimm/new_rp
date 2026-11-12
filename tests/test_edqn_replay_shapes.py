from __future__ import annotations
import math
import pickle
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import torch
from DQN.edqn import EDQNAgentRefactored, EDQNShapeConfig, UniformReplayMemoryNumpy


def legacy_scalar_fields(memory):
    fields = [(name, memory.memory.dtype[name].base) if name in ("action", "reward", "done") else (name, memory.memory.dtype[name]) for name in memory.memory.dtype.names]
    restored = np.empty(memory.memory.shape, dtype=fields)
    for name in restored.dtype.names:
        values = memory.memory[name]
        restored[name] = values.reshape(-1) if name in ("action", "reward", "done") else values
    memory.memory = restored
    return memory


def fill(memory, count=8, state_dim=4, actions=2):
    for i in range(count):
        memory.push(np.full(state_dim, i / 100, np.float32), np.array([i % actions], np.int64),
                    np.ones(state_dim, np.float32), np.array([i / 100], np.float32),
                    np.array([1.], np.float32), np.ones(actions, dtype=bool), None, None, None, None)


class EDQNReplayShapeTests(unittest.TestCase):
    def test_explicit_size_one_dtype_has_feature_axis_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = UniformReplayMemoryNumpy(8, 4, 1, 2)
        self.assertFalse([w for w in caught if issubclass(w.category, FutureWarning)])
        for name in ("action", "reward", "done"):
            self.assertEqual(m.memory.dtype[name].shape, (1,))

    def test_fresh_sample_shapes_and_dtypes(self):
        m = UniformReplayMemoryNumpy(8, 4, 1, 2); fill(m)
        batch = m.sample(4)
        self.assertEqual([tuple(x.shape) for x in batch[:6]], [(4, 4), (4, 1), (4, 4), (4, 1), (4, 1), (4, 2)])
        self.assertEqual(batch[1].dtype, torch.int64)
        self.assertEqual(batch[5].dtype, torch.bool)

    def test_pickled_scalar_replay_retains_contents_and_position(self):
        m = UniformReplayMemoryNumpy(8, 4, 1, 2); fill(m)
        m = pickle.loads(pickle.dumps(legacy_scalar_fields(m)))
        before = m.memory.copy(); position, full = m.position, m.full
        batch = m.sample(4)
        for idx in (1, 3, 4): self.assertEqual(tuple(batch[idx].shape), (4, 1))
        self.assertEqual((m.position, m.full, len(m)), (position, full, 8))
        np.testing.assert_array_equal(m.memory, before)

    def test_batch_size_one_preserves_batch_and_feature_axes(self):
        for legacy in (False, True):
            m = UniformReplayMemoryNumpy(8, 4, 1, 2); fill(m)
            if legacy: legacy_scalar_fields(m)
            batch = m.sample(1)
            for idx in (1, 3, 4): self.assertEqual(tuple(batch[idx].shape), (1, 1))

    def test_multidimensional_state_is_not_flattened(self):
        m = UniformReplayMemoryNumpy(4, (2, 2), (1,), 2)
        for _ in range(4):
            m.push(np.ones((2, 2)), np.array([0]), np.zeros((2, 2)), np.array([1]), np.array([0]), np.ones(2, dtype=bool), None, None, None, None)
        self.assertEqual(tuple(m.sample(2)[0].shape), (2, 2, 2))

    def test_real_new_env_update_128_samples_fresh_and_restored(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                torch.manual_seed(42)
                shape = EDQNShapeConfig(num_nodes=379, max_neighbors=4)
                agent = EDQNAgentRefactored(shape, batch_size=128, memory_capacity=128, device=torch.device("cpu"))
                fill(agent.memory, 128, state_dim=shape.dqn_input_dim, actions=shape.max_neighbors)
                if legacy: agent.memory = pickle.loads(pickle.dumps(legacy_scalar_fields(agent.memory)))
                parameter_before = next(agent.eval_net.parameters()).detach().clone()
                records = {}; original_sample = agent.memory.sample
                def sample(count):
                    batch = original_sample(count); records["reward"] = batch[3].clone(); return batch
                original_loss = torch.nn.functional.smooth_l1_loss
                def loss(prediction, target, **kwargs):
                    self.assertEqual(tuple(prediction.shape), (128, 1))
                    self.assertEqual(tuple(target.shape), (128, 1))
                    torch.testing.assert_close(target, records["reward"])
                    return original_loss(prediction, target, **kwargs)
                with patch.object(agent.memory, "sample", side_effect=sample), patch("DQN.edqn.F.smooth_l1_loss", side_effect=loss):
                    value = agent.learn_dqn_neighbor()
                self.assertTrue(math.isfinite(value))
                self.assertEqual(agent.learn_step_counter, 1)
                self.assertFalse(torch.equal(parameter_before, next(agent.eval_net.parameters()).detach()))


if __name__ == "__main__": unittest.main()
