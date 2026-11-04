from __future__ import annotations

import importlib.util
import sys
import types
import typing
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
_MISSING = object()


def load_episode_runner() -> types.ModuleType:
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object

    class FakeEVRuntimeState:
        def __init__(self, n_nodes: int):
            self.n_nodes = n_nodes
            self.done = False
            self.out = False
            self.total_cost = 0.0
            self.time_spend = 0.0
            self.routes = []
            self.failure_reason = None

    fake_runtime_state = types.ModuleType("rp.experiment.runtime_state")
    fake_runtime_state.EVRuntimeState = FakeEVRuntimeState
    fake_runtime_state.EpisodeLogger = object

    fake_graph_package = types.ModuleType("rp.graph")
    fake_graph_package.build_data_runtime = lambda *args, **kwargs: None
    fake_graph_package.dynamic_node_feature_update = lambda *args, **kwargs: None
    fake_graph_package.freeze_pyg_data = lambda value: value

    fake_decision_helpers = types.ModuleType("decision_helpers")
    fake_decision_helpers.action_mask_dqn = lambda *args, **kwargs: None

    injected = {
        "torch": fake_torch,
        "decision_helpers": fake_decision_helpers,
        "rp.experiment.runtime_state": fake_runtime_state,
        "rp.graph": fake_graph_package,
    }
    previous = {name: sys.modules.get(name, _MISSING) for name in injected}
    module_name = "rp.experiment._characterization_episode_runner"
    previous_loaded = sys.modules.get(module_name, _MISSING)
    try:
        sys.modules.update(injected)
        spec = importlib.util.spec_from_file_location(
            module_name,
            PROJECT_ROOT / "rp" / "experiment" / "episode_runner.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load rp/experiment/episode_runner.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_loaded is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_loaded
        for name, old_module in previous.items():
            if old_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


episode_runner = load_episode_runner()


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class SizedMemory:
    def __init__(self, size: int, complete_size: int | None = None):
        self.size = size
        self.complete_size = complete_size

    def __len__(self) -> int:
        return self.size


class EpisodePureFunctionCharacterizationTests(unittest.TestCase):
    def test_transition_annotations_describe_all_ten_payload_fields(self):
        for function in (
            episode_runner.make_terminal_transition,
            episode_runner.make_step_transition,
        ):
            with self.subTest(function=function.__name__):
                return_type = typing.get_type_hints(function)["return"]
                self.assertEqual(len(typing.get_args(return_type)), 10)

    def test_explicit_empty_ev_list_overrides_scenario_list(self) -> None:
        scenario_ids = ["EV_0", "EV_1"]

        self.assertIs(
            episode_runner.resolve_ev_ids(None, scenario_ids),
            scenario_ids,
        )
        self.assertEqual(episode_runner.resolve_ev_ids([], scenario_ids), [])

    def test_decision_distance_threshold_is_strict(self) -> None:
        self.assertTrue(episode_runner.should_make_decision(99.999, 100.0))
        self.assertFalse(episode_runner.should_make_decision(100.0, 100.0))
        self.assertFalse(episode_runner.should_make_decision(100.001, 100.0))

    def test_distance_to_lane_end_is_clamped_at_zero(self) -> None:
        traci = SimpleNamespace(
            vehicle=SimpleNamespace(
                getLaneID=lambda _ev_id: "lane-0",
                getLanePosition=lambda _ev_id: 120.0,
            ),
            lane=SimpleNamespace(getLength=lambda _lane_id: 100.0),
        )

        self.assertEqual(episode_runner.dist_to_lane_end(traci, "EV_0"), 0.0)

    def test_movement_out_lanes_match_exact_edge_ids(self) -> None:
        traci = SimpleNamespace(
            trafficlight=SimpleNamespace(
                getIDList=lambda: ["tls"],
                getControlledLinks=lambda _tls: [
                    [("edge10_0", "outbound_0", "via")],
                    [("edge1_0", "out_0", "via")],
                    [("edge1_1", "out_1", "via")],
                ],
            )
        )

        self.assertEqual(
            episode_runner.resolve_movement_out_lanes(traci, "edge1", "out"),
            ["out_0", "out_1"],
        )

    def test_step_transition_keeps_current_tuple_layout_and_dtypes(self) -> None:
        agent = SimpleNamespace(
            _pad_neighbors=lambda neighbors: (
                FakeTensor(neighbors),
                FakeTensor([True, False, True]),
            )
        )
        state = SimpleNamespace(
            last_state=FakeTensor([1.5, 2.5]),
            idx=1,
            transition={"agent": agent},
            data_t="old-graph",
            bag="old-bag",
        )

        payload = episode_runner.make_step_transition(
            state=state,
            next_state_tensor=FakeTensor([3.5, 4.5]),
            next_neighbors=[2, 4],
            reward=-1.25,
            done=0.0,
            next_data_t="new-graph",
            next_bag="new-bag",
        )

        # The current implementation returns ten items even though its return
        # annotation presently describes only nine. Characterization follows
        # the runtime payload consumed by replay memory.
        self.assertEqual(len(payload), 10)
        np.testing.assert_array_equal(payload[0], np.array([1.5, 2.5], dtype=np.float32))
        np.testing.assert_array_equal(payload[1], np.array([1], dtype=np.int64))
        np.testing.assert_array_equal(payload[2], np.array([3.5, 4.5], dtype=np.float32))
        np.testing.assert_array_equal(payload[3], np.array([-1.25], dtype=np.float32))
        np.testing.assert_array_equal(payload[4], np.array([0.0], dtype=np.float32))
        np.testing.assert_array_equal(payload[5], np.array([True, False, True], dtype=np.bool_))
        self.assertEqual(payload[6:], ("old-graph", "new-graph", "old-bag", "new-bag"))


class TransitionStorageCharacterizationTests(unittest.TestCase):
    def test_trajectory_memory_receives_per_ev_metadata(self) -> None:
        calls = []

        class TrajectoryMemory:
            def push_trajectory(self, *args, **kwargs):
                calls.append((args, kwargs))

        agent = SimpleNamespace(memory=TrajectoryMemory())
        state = SimpleNamespace(trajectory_id="rollout-1:EV_0", trajectory_t=2)

        episode_runner.push_agent_transition(
            agent=agent,
            state=state,
            ev_id="EV_0",
            payload=("state", "action", "next-state"),
            terminated=False,
            truncated=True,
        )

        self.assertEqual(calls[0][0], ("state", "action", "next-state"))
        self.assertEqual(
            calls[0][1],
            {
                "trajectory_key": "rollout-1:EV_0",
                "timestep": 2,
                "terminated": False,
                "truncated": True,
            },
        )
        self.assertEqual(state.trajectory_t, 3)

    def test_legacy_memory_uses_plain_push(self) -> None:
        calls = []
        memory = SimpleNamespace(push=lambda *payload: calls.append(payload))
        agent = SimpleNamespace(memory=memory)
        state = SimpleNamespace(trajectory_id="rollout-1:EV_0", trajectory_t=0)

        episode_runner.push_agent_transition(
            agent=agent,
            state=state,
            ev_id="EV_0",
            payload=(1, 2, 3),
            terminated=True,
        )

        self.assertEqual(calls, [(1, 2, 3)])
        self.assertEqual(state.trajectory_t, 1)

    def test_missing_trajectory_id_is_rejected(self) -> None:
        state = SimpleNamespace(trajectory_id=None, trajectory_t=0)
        agent = SimpleNamespace(memory=SimpleNamespace(push=lambda *_args: None))

        with self.assertRaisesRegex(RuntimeError, "EV_7"):
            episode_runner.push_agent_transition(
                agent=agent,
                state=state,
                ev_id="EV_7",
                payload=(),
                terminated=False,
            )


class ArrivalAndLearningCharacterizationTests(unittest.TestCase):
    def test_arrival_reward_uses_elapsed_minutes_bonus_and_shaping(self) -> None:
        state = SimpleNamespace(
            done=False,
            prev_decision_time=60.0,
            time_spend=1.0,
            total_cost=4.0,
            last_state=None,
            idx=None,
        )

        pushed_steps = episode_runner.handle_arrival(
            agent=SimpleNamespace(memory=SizedMemory(0)),
            train=True,
            evaluate=False,
            now=180.0,
            ev_id="EV_0",
            goal_node_id="goal",
            state=state,
            success_bonus=3.0,
            memory_push_learning_threshold=100,
            terminal_shaping_reward=0.5,
        )

        self.assertTrue(state.done)
        self.assertEqual(state.time_spend, 3.0)
        self.assertEqual(state.total_cost, 5.5)
        self.assertEqual(pushed_steps, 0)

    def test_learning_is_capped_at_64_updates_and_reports_counters(self) -> None:
        class Agent:
            def __init__(self):
                self.memory = SizedMemory(100)
                self.update_attempt_counter = 0
                self.learn_step_counter = 0
                self.gradient_step_counter = 0
                self.last_update_metrics = {}

            def learn_dqn_neighbor(self):
                self.update_attempt_counter += 1
                self.learn_step_counter += 1
                self.gradient_step_counter += 2
                self.last_update_metrics = {"critic_loss": 2.0}
                return 2.0

        agent = Agent()

        stats = episode_runner.learn_if_needed(
            agent,
            total_steps=100,
            train=True,
            evaluate=False,
            memory_learning_threshold=10,
        )

        self.assertEqual(agent.learn_step_counter, 64)
        self.assertEqual(stats["update_attempts"], 64)
        self.assertEqual(stats["learn_calls"], 64)
        self.assertEqual(stats["gradient_steps"], 128)
        self.assertEqual(stats["update_metrics"], {"critic_loss": 2.0})
        self.assertEqual(stats["last_loss"], 2.0)
        self.assertEqual(stats["mean_loss"], 2.0)

    def test_learning_error_is_wrapped_with_memory_context(self) -> None:
        class FailingAgent:
            def __init__(self):
                self.memory = SizedMemory(5, complete_size=3)

            def learn_dqn_neighbor(self):
                raise ValueError("bad batch")

        with self.assertRaisesRegex(
            episode_runner.AgentLearningError,
            "memory_size=5, complete_size=3",
        ):
            episode_runner.learn_if_needed(
                FailingAgent(),
                total_steps=1,
                train=True,
                evaluate=False,
                memory_learning_threshold=1,
            )


class RewardCharacterizationTests(unittest.TestCase):
    def make_runner(self, **config_overrides):
        config = episode_runner.EpisodeConfig(**config_overrides)
        runner = object.__new__(episode_runner.DQNGNNEpisodeRunner)
        runner.config = config
        runner.nid2idx = {"previous": 0, "current": 1}
        runner._reward_penalty_dead_end = float(config.dead_end_penalty)
        return runner

    def test_followup_reward_combines_uturn_penalty_and_elapsed_minutes(self) -> None:
        runner = self.make_runner()
        runner.nid2idx = {"current": 0}
        state = SimpleNamespace(
            recent_nodes=[0, 1, 0],
            prev_decision_time=60.0,
            time_spend=0.0,
            total_cost=5.0,
            prev_edge="old-edge",
        )

        reward = runner._compute_followup_reward(
            state,
            current_node="current",
            current_edge="new-edge",
            goal_node_id="goal",
            now=180.0,
        )

        self.assertEqual(reward, -3.0)
        self.assertEqual(state.time_spend, 2.0)
        self.assertEqual(state.total_cost, 2.0)

    def test_recent_revisit_penalty_is_applied_when_enabled(self) -> None:
        runner = self.make_runner(
            revisit_penalty_enabled=True,
            revisit_penalty=1.5,
        )
        runner.nid2idx = {"current": 0}
        state = SimpleNamespace(
            recent_nodes=[0, 1, 2, 0],
            prev_decision_time=60.0,
            time_spend=0.0,
            total_cost=0.0,
            prev_edge="old-edge",
        )

        reward = runner._compute_followup_reward(
            state,
            current_node="current",
            current_edge="new-edge",
            goal_node_id="goal",
            now=120.0,
        )

        self.assertEqual(reward, -2.5)

    def test_uturn_and_revisit_penalties_are_not_double_counted(self) -> None:
        runner = self.make_runner(
            uturn_penalty=2.0,
            revisit_penalty_enabled=True,
            revisit_penalty=5.0,
        )
        runner.nid2idx = {"current": 0}
        state = SimpleNamespace(
            recent_nodes=[0, 1, 0],
            prev_decision_time=60.0,
            time_spend=0.0,
            total_cost=0.0,
            prev_edge="old-edge",
        )

        reward = runner._compute_followup_reward(
            state,
            current_node="current",
            current_edge="new-edge",
            goal_node_id="goal",
            now=120.0,
        )

        self.assertEqual(reward, -3.0)

    def test_progress_reward_preserves_current_tie_and_regression_rules(self) -> None:
        runner = self.make_runner(
            progress_reward_enabled=True,
            progress_reward_alpha=0.5,
            progress_reward_beta=0.25,
        )

        with patch.object(runner, "_static_shortest_distance", side_effect=[100.0, 100.0]):
            self.assertEqual(runner._progress_reward("old", "new", "goal"), 0.5)
        with patch.object(runner, "_static_shortest_distance", side_effect=[100.0, 101.0]):
            self.assertEqual(runner._progress_reward("old", "new", "goal"), -0.25)

    def test_potential_shaping_uses_current_gamma_formula(self) -> None:
        runner = self.make_runner(
            potential_shaping_enabled=True,
            potential_shaping_gamma=0.99,
            potential_distance_scale=1000.0,
        )

        with patch.object(runner, "_static_shortest_distance", side_effect=[1000.0, 500.0]):
            self.assertAlmostEqual(
                runner._potential_shaping_reward("old", "new", "goal"),
                0.505,
            )
        with patch.object(runner, "_static_shortest_distance", return_value=1000.0):
            self.assertEqual(runner._terminal_potential_reward("old", "goal"), 1.0)

    def test_traffic_penalty_starts_at_threshold_and_scales_by_vehicle_count(self) -> None:
        runner = self.make_runner(
            traffic_penalty_enabled=True,
            traffic_penalty_threshold=4,
            traffic_penalty_delta=0.2,
        )
        counts = iter([3, 4])
        runner.traci = SimpleNamespace(
            edge=SimpleNamespace(getLastStepVehicleNumber=lambda _edge: next(counts))
        )

        self.assertEqual(runner._traffic_density_penalty("edge"), 0.0)
        self.assertEqual(runner._traffic_density_penalty("edge"), -0.8)

    def test_failure_penalty_is_terminal_and_idempotent(self) -> None:
        runner = self.make_runner(dead_end_penalty=20.0)
        runner.agent = SimpleNamespace(memory=SizedMemory(0))
        state = SimpleNamespace(
            done=False,
            out=False,
            prev_decision_time=60.0,
            prev_edge="old-edge",
            time_spend=0.0,
            total_cost=0.0,
            failure_reason=None,
            last_state=None,
            idx=None,
        )
        run_input = episode_runner.EpisodeRunInput(
            ev_ids=["EV_0"],
            goal_nodes={"EV_0": "goal"},
            train=True,
            evaluate=False,
        )

        self.assertEqual(
            runner._terminate_failure(
                "EV_0",
                state,
                run_input,
                180.0,
                reason="dead_end",
            ),
            0,
        )
        self.assertTrue(state.out)
        self.assertEqual(state.failure_reason, "dead_end")
        self.assertEqual(state.time_spend, 2.0)
        self.assertEqual(state.total_cost, -22.0)

        self.assertEqual(
            runner._terminate_failure(
                "EV_0",
                state,
                run_input,
                240.0,
                reason="second_failure",
            ),
            0,
        )
        self.assertEqual(state.failure_reason, "dead_end")
        self.assertEqual(state.total_cost, -22.0)


class EpisodeSummaryCharacterizationTests(unittest.TestCase):
    def test_summary_preserves_order_and_penalizes_failed_time_to_horizon(self) -> None:
        successful = SimpleNamespace(
            done=True,
            total_cost=3.0,
            time_spend=2.0,
            routes=["a", "b"],
            failure_reason=None,
        )
        failed = SimpleNamespace(
            done=False,
            total_cost=-20.0,
            time_spend=0.2,
            routes=["c"],
            failure_reason="dead_end",
        )
        states = {"EV_0": successful, "EV_1": failed}
        paths = []

        self.assertEqual(
            episode_runner.summarize_episode(states, paths, evaluate=True),
            [3.0, -20.0],
        )
        self.assertEqual(paths, [["a", "b"], ["c"]])

        runner = object.__new__(episode_runner.DQNGNNEpisodeRunner)
        runner.config = episode_runner.EpisodeConfig(
            max_simulation_steps=100,
            simulation_step_sec=0.5,
        )
        runtime = episode_runner.EpisodeRuntime(
            states=states,
            total_steps=7,
            simulation_steps=100,
            decisions={"EV_0": 2, "EV_1": 1},
        )
        stats = runner._summarize_runtime(
            runtime,
            learn_stats={"learn_calls": 4, "mean_loss": 1.25},
            bottleneck_stats={"bottleneck_triggered": True},
        )

        self.assertEqual(stats["episode_cost_minutes"], 2.2)
        self.assertEqual(stats["ev_costs_minutes"], [2.0, 0.2])
        self.assertAlmostEqual(stats["penalized_ev_costs_minutes"][1], 5.0 / 6.0)
        self.assertEqual(stats["success_count"], 1)
        self.assertEqual(stats["failure_count"], 1)
        self.assertEqual(stats["success_rate"], 0.5)
        self.assertEqual(stats["failure_reasons"], {"EV_1": "dead_end"})
        self.assertTrue(stats["truncated_by_max_steps"])
        self.assertEqual(stats["transition_steps"], 7)
        self.assertEqual(stats["learn_calls"], 4)
        self.assertEqual(stats["mean_loss"], 1.25)
        self.assertTrue(stats["bottleneck_triggered"])


if __name__ == "__main__":
    unittest.main()
