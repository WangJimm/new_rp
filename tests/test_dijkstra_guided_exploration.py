from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rp.experiment.episode_runner import (
    DQNGNNEpisodeRunner,
    EpisodeRunInput,
    EpisodeRuntime,
)
from rp.experiment.training import dijkstra_guidance_probability


class FakeNode:
    def __init__(self, node_id: str):
        self.node_id = node_id

    def getID(self) -> str:
        return self.node_id


class FakeEdge:
    def __init__(self, to_node: str):
        self.to_node = FakeNode(to_node)

    def getToNode(self) -> FakeNode:
        return self.to_node


class FakeNet:
    def getEdge(self, edge_id: str) -> FakeEdge:
        if edge_id != "start":
            raise KeyError(edge_id)
        return FakeEdge("A")


class FakeAgent:
    EPSILON = 1.0

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def choose_action_gnn(self, *args, evaluate: bool = False) -> int:
        self.calls.append(evaluate)
        return 2


def guidance_config(**overrides):
    values = {
        "edqn_dijkstra_guidance_enabled": True,
        "edqn_dijkstra_guide_probability": 0.7,
        "edqn_dijkstra_guide_hold_episodes": 500,
        "edqn_dijkstra_guide_decay_episodes": 1000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DijkstraGuidanceScheduleTests(unittest.TestCase):
    def test_probability_holds_then_linearly_decays_to_zero(self) -> None:
        config = guidance_config()

        self.assertEqual(dijkstra_guidance_probability(config, 0), 0.7)
        self.assertEqual(dijkstra_guidance_probability(config, 499), 0.7)
        self.assertEqual(dijkstra_guidance_probability(config, 1000), 0.35)
        self.assertEqual(dijkstra_guidance_probability(config, 1500), 0.0)
        self.assertEqual(dijkstra_guidance_probability(config, 9999), 0.0)

    def test_disabled_guidance_is_always_zero(self) -> None:
        config = guidance_config(edqn_dijkstra_guidance_enabled=False)
        self.assertEqual(dijkstra_guidance_probability(config, 0), 0.0)


class DijkstraGuidedActionTests(unittest.TestCase):
    def make_runner(self) -> DQNGNNEpisodeRunner:
        runner = object.__new__(DQNGNNEpisodeRunner)
        runner.net = FakeNet()
        runner.static_pack = {
            "adj": {
                "A": [("B", "ab"), ("C", "ac")],
                "B": [("G", "bg")],
                "C": [("G", "cg")],
            },
            "edge_len": {"ab": 1.0, "ac": 1.0, "bg": 1.0, "cg": 5.0},
        }
        runner.nid2idx = {"A": 0, "B": 1, "C": 2, "G": 3}
        runner.agent = FakeAgent()
        return runner

    @staticmethod
    def snapshot():
        return SimpleNamespace(
            current_edge="start",
            state_tensor="state",
            action_mask="mask",
            neighbors=[1, 2],
            bag=(0, 3, [1, 2], "tls"),
            graph_data="graph",
        )

    def test_teacher_replaces_an_epsilon_exploration_action(self) -> None:
        runner = self.make_runner()
        runtime = EpisodeRuntime.create([], n_nodes=4)
        run_input = EpisodeRunInput(
            ev_ids=[],
            goal_nodes={},
            dijkstra_guidance_probability=0.7,
        )

        with patch("rp.experiment.episode_runner.random.random", side_effect=[0.1, 0.1]):
            action = runner._choose_action(self.snapshot(), "G", run_input, runtime)

        self.assertEqual(action, 1)
        self.assertEqual(runtime.exploration_decision_count, 1)
        self.assertEqual(runtime.dijkstra_guidance_available_count, 1)
        self.assertEqual(runtime.dijkstra_guided_decision_count, 1)
        self.assertEqual(runner.agent.calls, [])

    def test_evaluation_always_uses_the_agent_without_teacher_guidance(self) -> None:
        runner = self.make_runner()
        runtime = EpisodeRuntime.create([], n_nodes=4)
        run_input = EpisodeRunInput(
            ev_ids=[],
            goal_nodes={},
            evaluate=True,
            train=False,
            dijkstra_guidance_probability=0.7,
        )

        action = runner._choose_action(self.snapshot(), "G", run_input, runtime)

        self.assertEqual(action, 2)
        self.assertEqual(runtime.dijkstra_guided_decision_count, 0)
        self.assertEqual(runner.agent.calls, [True])


if __name__ == "__main__":
    unittest.main()
