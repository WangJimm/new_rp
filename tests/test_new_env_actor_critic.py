from __future__ import annotations

import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from DQN.ppo_torch import TorchPPOAgentRefactored, TorchPPOShapeConfig
from DQN.sacd import SACDAgentRefactored, SACDShapeConfig, SACDPrioritizedReplayMemoryNumpy
from DQN.ppo_rllib import RLLibPPOAgentRefactored
from rp.experiment.config import build_selected_env_config
from rp.graph.features import get_edge_index_v2
from sumolib.net import readNet
from scripts import run_experiment as experiment


ROOT = Path(__file__).resolve().parents[1]


class NewEnvActorCriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        net = readNet(str(ROOT / "new_env/env_static_tls.net.xml"), withInternal=True)
        _, edge_ids, nid2idx, adj, edge_len = get_edge_index_v2(net)
        cls.pack = {"nid2idx": nid2idx, "adj": adj, "edge_len": edge_len,
                    "static_edge_attr_norm": torch.ones((len(edge_ids), 5))}

    def sac_agent(self):
        return SACDAgentRefactored(SACDShapeConfig(379, 4), batch_size=256,
                                  memory_capacity=256, device=torch.device("cpu"))

    def fill_sac(self, agent):
        for i in range(256):
            state = np.zeros(759, dtype=np.float32)
            state[i % 379] = 1.
            state[379 + (i + 1) % 379] = 1.
            state[-1] = i / 256
            count = 1 + i % 4
            action = i % count
            terminal = i % 7 == 0
            next_mask = np.zeros(4, dtype=bool)
            if not terminal:
                next_mask[:count] = True
            agent.memory.push(state, np.array([action]), state.copy(),
                              np.array([-20. if terminal else -.2], dtype=np.float32),
                              np.array([float(terminal)], dtype=np.float32), next_mask,
                              None, None, (0, 1, list(range(count)), torch.empty(count, 0)), None)

    def test_factories_use_new_env_shape_without_ray(self):
        with patch("torch.cuda.is_available", return_value=False):
            ppo = experiment.build_refactored_ppo_torch_agent_from_static_pack(self.pack)
            sac = experiment.build_refactored_sacd_agent_from_static_pack(self.pack)
            enriched = experiment.build_refactored_sacd_agent_from_static_pack(self.pack, use_action_features=True)
        self.assertEqual((ppo.num_nodes, ppo.a_size, ppo.shape.ppo_input_dim), (379, 4, 759))
        self.assertEqual((sac.num_nodes, sac.a_size, sac.shape.state_dim), (379, 4, 759))
        self.assertFalse(ppo.requires_graph_data)
        self.assertFalse(sac.requires_graph_data)
        self.assertTrue(enriched.requires_graph_data)
        self.assertEqual(enriched.shape.state_dim, 819)
        self.assertFalse(RLLibPPOAgentRefactored.requires_graph_data)

    def test_sac_replay_shapes_and_actual_updates(self):
        agent = self.sac_agent()
        self.fill_sac(agent)
        sample = agent.memory.sample(256)
        self.assertEqual([tuple(x.shape) for x in sample[:7]],
                         [(256, 759), (256, 1), (256, 759), (256, 1),
                          (256, 1), (256, 4), (256, 4)])
        before = agent.networks.q1.net[0].weight.detach().clone()
        for _ in range(3):
            self.assertTrue(np.isfinite(agent.learn_dqn_neighbor()))
        self.assertEqual(agent.learn_step_counter, 3)
        self.assertFalse(torch.equal(before, agent.networks.q1.net[0].weight))

    def test_legacy_scalar_sac_replay_load_and_update(self):
        agent = self.sac_agent()
        self.fill_sac(agent)
        original = agent.memory.memory
        scalar_fields = {"action", "reward", "done"}
        dtype = [(name, original.dtype.fields[name][0].base if name in scalar_fields
                  else original.dtype.fields[name][0]) for name in original.dtype.names]
        legacy = np.zeros(256, dtype=dtype)
        for name in original.dtype.names:
            legacy[name] = original[name].reshape(256) if name in scalar_fields else original[name]
        payload = agent.memory.state_dict()
        payload["memory"] = legacy
        agent.memory.load_state_dict(payload)
        self.assertEqual(tuple(agent.memory.sample(256)[1].shape), (256, 1))
        self.assertTrue(np.isfinite(agent.learn_dqn_neighbor()))

    def test_size_one_sac_masks_retain_axis(self):
        memory = SACDPrioritizedReplayMemoryNumpy(4, 1, 1, 1)
        for _ in range(4):
            memory.push(np.ones(1), np.zeros(1), np.ones(1), np.ones(1),
                        np.ones(1), np.ones(1, dtype=bool))
        self.assertTrue(all(tuple(x.shape) == (4, 1) for x in memory.sample(4)[:7]))

    def test_sac_checkpoint_roundtrip_keeps_policy_and_replay(self):
        agent = self.sac_agent()
        self.fill_sac(agent)
        agent.learn_dqn_neighbor()
        state = torch.from_numpy(agent.memory.memory["state"][1].copy())
        expected = agent.choose_action_gnn(state, None, [3, 5], 0, 1, None, evaluate=True)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "sac.pt")
            agent.save_tmp_file(12, path)
            restored = self.sac_agent()
            self.assertEqual(restored.load_full_model(filename=path), 12)
            self.assertEqual(len(restored.memory), 256)
            self.assertEqual(restored.choose_action_gnn(state, None, [3, 5], 0, 1, None, evaluate=True), expected)
            self.assertTrue(np.isfinite(restored.learn_dqn_neighbor()))

    def test_ppo_large_terminal_returns_train_the_value_head(self):
        agent = TorchPPOAgentRefactored(TorchPPOShapeConfig(379, 4), train_batch_size=8,
            sgd_minibatch_size=4, num_sgd_iter=2, device=torch.device("cpu"))
        for i in range(8):
            state = np.zeros(759, dtype=np.float32)
            state[i] = 1.
            state[379 + 1] = 1.
            agent.memory.push_trajectory(state, np.array([0]), state.copy(),
                np.array([-20.], dtype=np.float32), np.ones(1), np.zeros(4, dtype=bool),
                None, None, (0, 1, [1], None), None,
                trajectory_key=f"episode:EV_{i}", timestep=0, terminated=True)
        before = agent.network.value_head.weight.detach().clone()
        loss = agent.learn_dqn_neighbor()
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, agent.network.value_head.weight))
        self.assertGreater(agent.last_update_metrics["value_loss"], 10.)
        self.assertEqual(agent.memory.complete_size, 0)
        self.assertEqual(agent.gradient_step_counter, 4)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "ppo.pt")
            agent.save_tmp_file(3, path)
            restored = TorchPPOAgentRefactored(TorchPPOShapeConfig(379, 4), device=torch.device("cpu"))
            self.assertEqual(restored.load_full_model(filename=path), 3)
            for name, value in agent.network.state_dict().items():
                torch.testing.assert_close(value, restored.network.state_dict()[name], rtol=0, atol=0)

    def test_policy_actions_respect_local_masks(self):
        ppo = TorchPPOAgentRefactored(TorchPPOShapeConfig(379, 4), device=torch.device("cpu"))
        sac = self.sac_agent()
        for agent in [ppo, sac]:
            for count in [1, 2, 4]:
                neighbors = [11, 22, 33, 44][:count]
                for evaluate in [False, True]:
                    for _ in range(5):
                        self.assertIn(agent.choose_action_gnn(torch.zeros(759), None,
                            neighbors, 0, 1, None, evaluate=evaluate), neighbors)

    def test_cli_selects_budgets_without_changing_defaults(self):
        config = build_selected_env_config(ROOT, "new_env")
        self.assertIs(experiment.apply_episode_limit_args(config, argparse.Namespace()), config)
        for variant in ["ppo_torch_baseline", "sacd_baseline"]:
            with patch("sys.argv", ["run_experiment", "--env", "new_env", "--rp_variant", variant,
                                   "--max-decisions-per-ev", "80", "--max-simulation-steps", "6000"]):
                args = experiment.parse_args()
            result = experiment.apply_episode_limit_args(config, args)
            self.assertEqual((result.max_decisions_per_ev, result.max_simulation_steps), (80, 6000))
            self.assertEqual(args.rp_variant, variant)
        for field in ["max_decisions_per_ev", "max_simulation_steps"]:
            for bad in [0, -1]:
                with self.assertRaises(ValueError):
                    experiment.apply_episode_limit_args(config, argparse.Namespace(**{field: bad}))


if __name__ == "__main__":
    unittest.main()
