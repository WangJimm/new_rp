from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from DQN.dqn_re import DQNAgentRefactored, DQNShapeConfig
from rp.experiment.config import build_selected_env_config
from rp.experiment.runtime_state import EVRuntimeState
from rp.experiment import episode_runner as er
from scripts import run_experiment as run
from tests.test_replay_shapes import scalar_legacy_memory


class DQNEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.original_threads)

    def test_dynamic_factory_and_real_replay_updates_for_both_networks(self):
        for n, expected in [(115,238),(379,766)]:
            with self.subTest(nodes=n),patch('torch.cuda.is_available',return_value=False):
                agent=run.build_refactored_dqn_agent_from_static_pack(
                    {'nid2idx':{str(i):i for i in range(n)},'adj':{'source':[1,2,3,4]}})
                self.assertEqual(agent.shape.dqn_input_dim,expected)
                self.assertFalse(agent.requires_graph_data)
                self.assertEqual(agent.eval_net.fc1.in_features,expected)
                self.fill(agent)
                for legacy in [False,True]:
                    if legacy: scalar_legacy_memory(agent.memory)
                    batch=agent.memory.sample(64)
                    self.assertEqual([tuple(t.shape) for t in batch[:6]],
                        [(64,expected),(64,1),(64,expected),(64,1),(64,1),(64,4)])
                    before=agent.eval_net.fc1.weight.detach().clone()
                    self.assertTrue(np.isfinite(agent.learn_dqn_neighbor()))
                    self.assertFalse(torch.equal(before,agent.eval_net.fc1.weight))
                self.assertEqual(agent.learn_step_counter,2)

    @staticmethod
    def fill(agent):
        n=agent.num_nodes
        for i in range(64):
            base=torch.zeros(2*n); base[i%n]=1; base[n+(i+1)%n]=1
            state=agent.concat_state_tls(base,[[.25,.5],[.1,.0]])
            agent.memory.push(state.cpu().numpy(),np.array([i%2]),state.cpu().numpy(),
                np.array([-1.]),np.array([i%3==0]),np.array([True,True,False,False]),
                None,None,(0,1,[1,2],None),(0,1,[1,2],None))

    def observation_setup(self):
        agent=DQNAgentRefactored(DQNShapeConfig(3,2),memory_capacity=128,device=torch.device('cpu'))
        edges=[NS(getID=lambda:'ab',getToNode=lambda:NS(getID=lambda:'B')),
               NS(getID=lambda:'ac',getToNode=lambda:NS(getID=lambda:'C'))]
        net=NS(getEdge=lambda e:NS(getToNode=lambda:NS(getID=lambda:'A')))
        traci=NS(vehicle=NS(getRoadID=lambda ev:'in_A'))
        return agent,edges,net,traci

    def test_tls_survives_graph_skip_with_identical_states_masks_and_actions(self):
        agent,edges,net,traci=self.observation_setup()
        with ExitStack() as stack:
            stack.enter_context(patch.object(er,'candidate_next_junctions',return_value=(['B','C'],edges)))
            stack.enter_context(patch.object(er,'action_mask_dqn',return_value=torch.tensor([False,True,True])))
            tls=stack.enter_context(patch.object(er,'build_tls_state',return_value=[[.25,.5],[.1,.0]]))
            graph=stack.enter_context(patch.object(er,'build_data_runtime',return_value=Data(
                x=torch.ones(3,3),edge_index=torch.tensor([[0,1],[1,2]]),edge_attr=torch.ones(2,2))))
            stack.enter_context(patch.object(er,'dynamic_node_feature_update',side_effect=lambda g,t,r,v,a:(g,v,a)))
            features=stack.enter_context(patch.object(er,'build_candidate_action_features',return_value=torch.ones(2,4)))
            def snap(state):
                return er.build_agent_snapshot(traci=traci,net=net,static_pack={'nid2idx':{'A':0,'B':1,'C':2}},
                    travel_time_ema=None,mean_speed_ema=None,occupancy_ema=None,
                    agent=agent,state=state,ev_id='EV_0',goal_node_id='C')
            agent.requires_graph_data=True
            full=snap(EVRuntimeState(3))
            agent.requires_graph_data=False
            graph.reset_mock(); features.reset_mock()
            state=EVRuntimeState(3); fast=snap(state)
            graph.assert_not_called(); features.assert_not_called()
            self.assertEqual(tls.call_count,2)
            torch.testing.assert_close(fast.state_tensor,full.state_tensor,rtol=0,atol=0)
            torch.testing.assert_close(fast.state_tensor[-4:],torch.tensor([.25,.5,.1,0.]))
            torch.testing.assert_close(fast.action_mask,full.action_mask,rtol=0,atol=0)
            choose=lambda s:agent.choose_action_gnn(s.state_tensor,s.action_mask,s.neighbors,0,2,s.graph_data,evaluate=True)
            self.assertEqual(choose(fast),choose(full))
            self.assertIsNone(fast.graph_data)
            self.assertEqual(list(state.recent_nodes),[0])
            state.last_state=fast.state_tensor; state.bag=fast.bag; state.idx=0
            state.neighbors=fast.neighbors; state.transition['agent']=agent
            terminal=er.make_terminal_transition(agent=agent,state=state,reward=-20.,goal_node_id='C',done=1.)
            self.assertEqual(terminal[2].shape,(10,))
            np.testing.assert_array_equal(terminal[2][3:6],[0.,0.,1.])
            np.testing.assert_array_equal(terminal[2][-4:],np.zeros(4))
            self.assertIsNone(terminal[6]); self.assertIsNone(terminal[7])

    def test_checkpoint_roundtrip_and_legal_masked_action(self):
        for n in [115,379]:
            agent=DQNAgentRefactored(DQNShapeConfig(n,4),memory_capacity=64,device=torch.device('cpu'))
            self.fill(agent); agent.learn_dqn_neighbor()
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'dqn.pt'; agent.save_tmp_file(42,str(path))
                restored=DQNAgentRefactored(agent.shape,memory_capacity=64,device=torch.device('cpu'))
                self.assertEqual(restored.load_full_model(filename=str(path)),42)
                for k,v in agent.networks.state_dict().items():
                    torch.testing.assert_close(restored.networks.state_dict()[k],v,rtol=0,atol=0)
                x=agent.concat_state_tls(torch.zeros(2*n),[[.1,.2]])
                self.assertEqual(restored.choose_action_gnn(x,torch.zeros(n),[7],0,1,None,evaluate=True),7)

    def test_graph_payload_removal_preserves_actual_updates_and_priorities(self):
        shape=DQNShapeConfig(115,4)
        a=DQNAgentRefactored(shape,memory_capacity=64,device=torch.device('cpu'))
        b=DQNAgentRefactored(shape,memory_capacity=64,device=torch.device('cpu'))
        b.networks.load_state_dict(a.networks.state_dict())
        self.fill(a); self.fill(b)
        graph=Data(x=torch.ones(3,3),edge_index=torch.tensor([[0,1],[1,2]]),edge_attr=torch.ones(2,2))
        for i in range(64):
            a.memory.memory['data_t'][i]=graph
            a.memory.memory['next_data_t'][i]=graph
        for step in range(3):
            np.random.seed(42+step); loss_a=a.learn_dqn_neighbor()
            np.random.seed(42+step); loss_b=b.learn_dqn_neighbor()
            self.assertEqual(loss_a,loss_b)
            for k,v in a.networks.state_dict().items():
                torch.testing.assert_close(v,b.networks.state_dict()[k],rtol=0,atol=0)
            np.testing.assert_array_equal(a.memory.priorities,b.memory.priorities)

    def test_both_environments_resolve_their_own_dqn_checkpoints(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'THIS_DIR',Path(tmp)):
            for name in ['env','new_env']:
                config=replace(build_selected_env_config(root,name),rp_variant='dqn_baseline')
                self.assertIn('use_gnn=False',run.describe_runtime(config))
                default=run.default_evaluation_checkpoint(config)
                self.assertEqual(default.name,'dqn_best_arrival.pt')
                self.assertNotIn('dqngnn',str(default))
                strict=default.with_name('dqn_best_100pct_arrival_lowest_cost.pt')
                strict.parent.mkdir(parents=True,exist_ok=True); strict.touch()
                self.assertEqual(run.default_evaluation_checkpoint(config),strict)


if __name__=='__main__': unittest.main()
