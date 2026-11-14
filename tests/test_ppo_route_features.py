from __future__ import annotations
from dataclasses import replace
import os
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from DQN.ppo_route import RoutePPOAgent, RoutePPOShapeConfig, RouteActorCritic, ROUTE_FEATURE_NAMES
from rp.experiment.runtime_state import EVRuntimeState
from rp.experiment.reproducibility import seed_everything, capture_rng_state, restore_rng_state


class Node:
    def __init__(self, id, xy): self.id, self.xy = id, xy
    def getID(self): return self.id
    def getCoord(self): return self.xy


class Edge:
    def __init__(self, id, a, b, outgoing=()):
        self.id, self.a, self.b, self.outgoing = id, a, b, outgoing
    def getID(self): return self.id
    def getFromNode(self): return self.a
    def getToNode(self): return self.b
    def getOutgoing(self): return self.outgoing
    def allows(self, vclass): return True
    def getLength(self): return 50.


def fixture():
    nodes = {n: Node(n, xy) for n, xy in [('a',(0,0)),('b',(1,0)),('g',(2,0)),('d',(1,1))]}
    edges = {}
    for id, a, b in [('ab','a','b'),('ba','b','a'),('bg','b','g'),('bd','b','d')]:
        edges[id] = Edge(id, nodes[a], nodes[b])
    edges['ab'].outgoing = [edges['ba'], edges['bg'], edges['bd']]
    edges['ba'].outgoing = [edges['ab']]
    net = SimpleNamespace(getNode=nodes.__getitem__, getEdge=edges.__getitem__, getEdges=lambda:list(edges.values()))
    pack = {'nid2idx': {n:i for i,n in enumerate(nodes)}, 'edge_ids':list(edges)}
    return nodes, edges, net, pack


class RouteFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_shapes_and_no_graph_requirement(self):
        self.assertEqual(RoutePPOShapeConfig(115,257,4).ppo_input_dim, 540)
        self.assertEqual(RoutePPOShapeConfig(379,784,4).ppo_input_dim, 1595)
        self.assertFalse(RoutePPOAgent.requires_graph_data)

    def test_candidate_order_equivariance_and_masked_critic(self):
        shape = RoutePPOShapeConfig(4,4,3)
        network = RouteActorCritic(shape)
        x = torch.randn(2,shape.ppo_input_dim)
        masks = torch.tensor([[1,1,0],[1,0,1]], dtype=torch.bool)
        perm = torch.tensor([2,0,1])
        xp = torch.cat([x[:,:shape.context_dim], x[:,shape.context_dim:].reshape(2,3,12)[:,perm].flatten(1)],1)
        logits, values = network(x,masks)
        logits_p, values_p = network(xp,masks[:,perm])
        torch.testing.assert_close(logits_p, logits[:,perm])
        torch.testing.assert_close(values_p, values)
        self.assertTrue(torch.isfinite(values).all())

    def test_dead_end_filter_goal_exception_history_and_budget(self):
        nodes, edges, net, pack = fixture()
        shape = RoutePPOShapeConfig(4,4,3)
        agent = RoutePPOAgent(shape,device=torch.device('cpu'))
        agent.configure_route_context(net,pack,80)
        state = EVRuntimeState(4)
        traci = SimpleNamespace(edge=SimpleNamespace(getTraveltime=lambda e:20.))
        args = dict(base_state=torch.zeros(9), traci=traci, cur_edge='ab', goal_node_id='g', state=state,
                    next_junctions=['a','g','d'], outgoing_edges=edges['ab'].outgoing)
        x, next_nodes, kept, features = agent.prepare_route_snapshot(**args)
        self.assertEqual(next_nodes,['a','g'])
        self.assertEqual([e.getID() for e in kept],['ba','bg'])
        self.assertEqual(features[1,9],1.)
        self.assertEqual(x.shape, (shape.ppo_input_dim,))
        state.decisions = 40
        y, _, _, _ = agent.prepare_route_snapshot(**args)
        self.assertEqual(y[9+4].item(),.5)
        self.assertGreater(y[9+4+3].item(), x[9+4+3].item())
        self.assertEqual(agent.concat_state_action_features(torch.zeros(9)).numel(), shape.ppo_input_dim)

    def test_actual_ppo_update_and_state_reload(self):
        shape = RoutePPOShapeConfig(4,4,3)
        agent = RoutePPOAgent(shape,device=torch.device('cpu'))
        agent.train_batch_size = 8
        agent.sgd_minibatch_size = 4
        agent.target_kl = 1e-12
        before = {k:v.clone() for k,v in agent.network.state_dict().items()}
        for t in range(8):
            x = np.random.randn(shape.ppo_input_dim).astype(np.float32)
            agent.memory.push_trajectory(x, np.array([t%2]), x,
                np.array([1.]), np.array([t==7]), np.array([1.,1.,0.]),
                bag=(0,1,[0,1],None), trajectory_key='EV_0', timestep=t, terminated=t==7)
        self.assertTrue(np.isfinite(agent.learn_dqn_neighbor()))
        self.assertEqual(agent.learn_step_counter,1)
        self.assertGreater(agent.gradient_step_counter,0)
        self.assertLess(agent.last_update_metrics['epochs_completed'],agent.num_sgd_iter)
        self.assertEqual(agent.last_update_metrics['kl_early_stop'],1.)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in agent.network.state_dict().items()))
        restored = RoutePPOAgent(shape,device=torch.device('cpu'))
        restored.load_training_state(agent.get_training_state())
        for k,v in restored.network.state_dict().items(): torch.testing.assert_close(v, agent.network.state_dict()[k])

    def test_seed_and_rng_snapshot_restore(self):
        with patch.dict(os.environ, {'PYTHONHASHSEED':'42'}): seed_everything(42, deterministic=True)
        a = RouteActorCritic(RoutePPOShapeConfig(4,4,3))
        with patch.dict(os.environ, {'PYTHONHASHSEED':'42'}): seed_everything(42, deterministic=True)
        b = RouteActorCritic(RoutePPOShapeConfig(4,4,3))
        for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]))
        state = capture_rng_state()
        draws = random.random(), np.random.rand(), torch.rand(3)
        restore_rng_state(state)
        self.assertEqual(draws[0], random.random())
        self.assertEqual(draws[1], np.random.rand())
        self.assertTrue(torch.equal(draws[2], torch.rand(3)))

    def test_parallel_edge_keeps_viable_alternative(self):
        nodes, edges, net, pack = fixture()
        alternative = Edge('ba_trap', nodes['b'], nodes['a'])
        edges['ba_trap'] = alternative
        pack['edge_ids'].append('ba_trap')
        agent = RoutePPOAgent(RoutePPOShapeConfig(4,5,3),device=torch.device('cpu'))
        agent.configure_route_context(net,pack,80)
        _, next_nodes, kept, _ = agent.prepare_route_snapshot(base_state=torch.zeros(9),
            traci=SimpleNamespace(edge=SimpleNamespace(getTraveltime=lambda e:1. if e=='ba_trap' else 20.)),
            cur_edge='ab',goal_node_id='g',state=EVRuntimeState(4),
            next_junctions=['a'],outgoing_edges=[alternative,edges['ba']])
        self.assertEqual(next_nodes,['a'])
        self.assertEqual(kept,[edges['ba']])

    def test_cap_allows_goal_edge_only_for_new_variant(self):
        from rp.experiment.episode_runner import DQNGNNEpisodeRunner
        _, edges, net, _ = fixture()
        runner = object.__new__(DQNGNNEpisodeRunner)
        runner.net = net
        runner.config = SimpleNamespace(max_decisions_per_ev=80)
        runner.agent = SimpleNamespace(uses_route_features=True)
        state = EVRuntimeState(4,current_exit_edge='bg')
        runtime = SimpleNamespace(states={'EV_0':state},decisions={'EV_0':80},total_steps=0)
        run_input = SimpleNamespace(goal_nodes={'EV_0':'g'})
        runner._ready_for_decision = lambda ev:True
        with patch.object(runner,'_step_single_ev') as step, patch.object(runner,'_terminate_failure',return_value=1) as fail:
            runner._step_active_evs({'EV_0'},set(),runtime,run_input,100.)
            step.assert_called_once(); fail.assert_not_called()
        runner.agent.uses_route_features = False
        with patch.object(runner,'_step_single_ev') as step, patch.object(runner,'_terminate_failure',return_value=1) as fail:
            runner._step_active_evs({'EV_0'},set(),runtime,run_input,100.)
            step.assert_not_called(); fail.assert_called_once()

    def test_full_training_state_restores_rng_and_checks_seed(self):
        from rp.experiment import build_selected_env_config
        from scripts import run_experiment as run
        config = replace(build_selected_env_config(Path(__file__).resolve().parents[1],'env'),
                         rp_variant='ppo_route_features',training_seed=42,deterministic=True)
        agent = RoutePPOAgent(RoutePPOShapeConfig(4,4,3),device=torch.device('cpu'))
        bundle = SimpleNamespace(config=config,agent=agent,rp_variant='ppo_route_features',
                                 agent_type='ppo_route',edge_feature_mode=None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            run.save_training_state(bundle=bundle,artifacts=run.create_training_artifacts(),progress=run.TrainingProgress(),
                scenario_schedule=[],scenario_schedule_seed=42,next_episode=3,output_path=path)
            draws = random.random(), np.random.rand(), torch.rand(3)
            random.random(); np.random.rand(); torch.rand(3)
            run.load_training_state(bundle,path)
            self.assertEqual(draws[0],random.random())
            self.assertEqual(draws[1],np.random.rand())
            self.assertTrue(torch.equal(draws[2],torch.rand(3)))
            bundle.config = replace(config,training_seed=43)
            with self.assertRaisesRegex(ValueError,'training_seed'):
                run.load_training_state(bundle,path)

    def test_validation_restores_training_rng_even_on_exception(self):
        from scripts import run_experiment as run
        from unittest.mock import Mock
        bundle = SimpleNamespace(config=SimpleNamespace(training_seed=42),agent=SimpleNamespace(eval_net=Mock()))
        before = capture_rng_state()
        expected = random.random(), np.random.rand(), torch.rand(3)
        restore_rng_state(before)
        def failing_validation():
            random.random(); np.random.rand(); torch.rand(5)
            raise RuntimeError('validation error')
        with self.assertRaises(RuntimeError): run.run_with_agent_eval_mode(bundle,failing_validation)
        self.assertEqual(expected[0],random.random())
        self.assertEqual(expected[1],np.random.rand())
        self.assertTrue(torch.equal(expected[2],torch.rand(3)))

    def test_validation_and_testing_accept_seed_and_budget_flags(self):
        import sys
        from scripts import validation, evaluate
        for entrypoint in (validation,evaluate):
            with patch.object(sys,'argv',['entrypoint','--env','env','--rp_variant','ppo_route_features',
                '--training-seed','42','--deterministic','--max-decisions-per-ev','80','--max-simulation-steps','6000']):
                args = entrypoint.parse_args()
            self.assertEqual(args.training_seed,42)
            self.assertEqual(args.max_decisions_per_ev,80)
            self.assertTrue(args.deterministic)


if __name__ == '__main__': unittest.main()
