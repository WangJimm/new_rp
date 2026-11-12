from __future__ import annotations
import unittest
from collections import deque
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.data import Data
from DQN.edqn import EDQNAgentRefactored, EDQNShapeConfig
from rp.experiment.runtime_state import EVRuntimeState
import rp.experiment.episode_runner as er


class EDQNFastPathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.agent = EDQNAgentRefactored(EDQNShapeConfig(3, 2), memory_capacity=16, device=torch.device("cpu"))
        self.current = ["A"]
        self.clock = [120.]
        self.nids = {"A": 0, "B": 1, "C": 2}
        def edge(node):
            return NS(getID=lambda: "to_" + node, getToNode=lambda: NS(getID=lambda: node))
        self.edges = [edge("B"), edge("C")]
        self.net = NS(getEdge=lambda name: NS(getToNode=lambda: NS(getID=lambda: name.removeprefix("in_"))))
        self.traci = NS(vehicle=NS(getRoadID=lambda ev: "in_" + self.current[0]), simulation=NS(getTime=lambda: self.clock[0]))
        self.pack = {"nid2idx": self.nids}
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(er, "candidate_next_junctions", return_value=(["B", "C"], self.edges)))
        self.stack.enter_context(patch.object(er, "action_mask_dqn", return_value=torch.tensor([False, True, True])))
        self.graph_mock = self.stack.enter_context(patch.object(er, "build_data_runtime", side_effect=self.graph))
        self.features_mock = self.stack.enter_context(patch.object(er, "build_candidate_action_features", return_value=torch.ones((2, 15))))
        self.addCleanup(self.stack.close)

    @staticmethod
    def graph(*args, **kwargs):
        return Data(x=torch.ones((3, 3)), edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.ones((2, 8)))

    def snapshot(self, state):
        return er.build_agent_snapshot(traci=self.traci, net=self.net, static_pack=self.pack,
            travel_time_ema=None, mean_speed_ema=None, occupancy_ema=None,
            agent=self.agent, state=state, ev_id="EV_0", goal_node_id="C")

    def test_fast_path_matches_state_masks_and_greedy_action_without_graph_calls(self):
        self.agent.requires_graph_data = True
        full = self.snapshot(EVRuntimeState(3))
        self.agent.requires_graph_data = False
        self.graph_mock.reset_mock(); self.features_mock.reset_mock()
        fast = self.snapshot(EVRuntimeState(3))
        self.graph_mock.assert_not_called(); self.features_mock.assert_not_called()
        self.assertIsNone(fast.graph_data)
        self.assertEqual(fast.bag[:3], full.bag[:3])
        self.assertEqual(fast.bag[3].numel(), 0)
        torch.testing.assert_close(fast.state_tensor, full.state_tensor, rtol=0, atol=0)
        torch.testing.assert_close(fast.action_mask, full.action_mask, rtol=0, atol=0)
        self.assertEqual(tuple(fast.state_tensor.shape), (7,))
        self.agent.EPSILON = 0.
        choose = lambda snap: self.agent.choose_action_gnn(snap.state_tensor, snap.action_mask,
            snap.neighbors, snap.bag[0], snap.bag[1], snap.graph_data, evaluate=True)
        self.assertEqual(choose(fast), choose(full))

    def test_graph_agents_keep_the_full_path(self):
        self.agent = NS(num_nodes=3, concat_state_time=self.agent.concat_state_time)
        result = self.snapshot(EVRuntimeState(3))
        self.assertIsNotNone(result.graph_data)
        self.graph_mock.assert_called_once(); self.features_mock.assert_called_once()

    def test_recent_history_and_uturn_rewards_match_the_full_path(self):
        full_state, fast_state = EVRuntimeState(3), EVRuntimeState(3)
        runner = er.DQNGNNEpisodeRunner(traci=self.traci, net=self.net, static_pack=self.pack,
            agent=self.agent, travel_time_ema=None, mean_speed_ema=None, occupancy_ema=None,
            config=er.EpisodeConfig(revisit_penalty_enabled=True, revisit_penalty=1., uturn_penalty=1.))
        for step, node in enumerate(["A", "B", "A", "B", "A"], start=1):
            self.current[0] = node; self.clock[0] = float(step)
            self.agent.requires_graph_data = True; self.snapshot(full_state)
            self.agent.requires_graph_data = False; self.snapshot(fast_state)
            self.assertEqual(list(full_state.recent_nodes), list(fast_state.recent_nodes))
            for state in (full_state, fast_state): state.prev_decision_time = step - 1
            a = runner._compute_followup_reward(full_state, node, "in_" + node, "C", float(step))
            b = runner._compute_followup_reward(fast_state, node, "in_" + node, "C", float(step))
            self.assertEqual(a, b)
            if step >= 3: self.assertLess(b, -1.)
        self.assertEqual(list(fast_state.recent_nodes), [1, 0, 1, 0])

    def test_replay_omits_graph_payload_and_keeps_terminal_goal(self):
        state = EVRuntimeState(3); snap = self.snapshot(state)
        state.last_state = snap.state_tensor; state.data_t = snap.graph_data
        state.bag = snap.bag; state.idx = 0; state.neighbors = snap.neighbors
        state.transition["agent"] = self.agent
        with patch.object(er, "freeze_pyg_data", side_effect=AssertionError("unused graph copy")):
            transition = er.make_step_transition(state=state, next_state_tensor=snap.state_tensor,
                next_neighbors=snap.neighbors, reward=-.5, done=0., next_data_t=None, next_bag=snap.bag)
            terminal = er.make_terminal_transition(agent=self.agent, state=state, reward=3.,
                goal_node_id="C", done=1., current_time_sec=120.)
        self.assertIsNone(transition[6]); self.assertIsNone(transition[7])
        self.assertIsNone(terminal[6]); self.assertIsNone(terminal[7])
        np.testing.assert_array_equal(terminal[2][3:6], [0., 0., 1.])

    def test_three_real_updates_match_graph_and_slim_replays(self):
        shape = EDQNShapeConfig(379, 4)
        a = EDQNAgentRefactored(shape, batch_size=128, memory_capacity=128, device=torch.device("cpu"))
        b = EDQNAgentRefactored(shape, batch_size=128, memory_capacity=128, device=torch.device("cpu"))
        b.networks.load_state_dict(a.networks.state_dict())
        graph = self.graph()
        for i in range(128):
            state = np.zeros(shape.dqn_input_dim, dtype=np.float32)
            state[i % 379] = 1.; state[379 + (i + 1) % 379] = 1.; state[-1] = i / 128
            next_state = np.roll(state, 1)
            numeric = (state, np.array([i % 4]), next_state, np.array([i / 100], dtype=np.float32),
                       np.array([float(i % 3 == 0)], dtype=np.float32), np.ones(4, dtype=bool))
            a.memory.push(*numeric, graph, graph, (0, 2, [1, 2], torch.ones(2, 15)), (0, 2, [1, 2], torch.ones(2, 15)))
            b.memory.push(*numeric, None, None, (0, 2, [1, 2], torch.empty(2, 0)), (0, 2, [1, 2], torch.empty(2, 0)))
        for step in range(3):
            np.random.seed(100 + step); first = a.learn_dqn_neighbor()
            np.random.seed(100 + step); second = b.learn_dqn_neighbor()
            self.assertEqual(first, second)
            for name, parameter in a.networks.state_dict().items():
                torch.testing.assert_close(parameter, b.networks.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__": unittest.main()
