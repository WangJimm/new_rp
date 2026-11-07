from __future__ import annotations
import math
import pickle
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.data import Data
from DQN.dqngnn_re import PrioritizedReplayMemoryNumpy
from DQN.dqngnn_action_re import DQNGNNActionAgentRefactored, DQNGNNActionShapeConfig


def scalar_legacy_memory(memory):
    """Recreate the scalar dtype emitted by affected NumPy versions/checkpoints."""
    fields=[]
    for name in memory.memory.dtype.names:
        field=memory.memory.dtype[name]
        if name in ('action','reward','done'):
            fields.append((name,field.base))
        else:
            fields.append((name,field))
    old=np.empty(memory.memory.shape,dtype=fields)
    for name in memory.memory.dtype.names:
        values=memory.memory[name]
        old[name]=values.reshape(len(values)) if name in ('action','reward','done') else values
    memory.memory=old
    return memory


def fill_memory(memory, count=8, graph=False):
    for i in range(count):
        data=Data(x=torch.tensor([[1.,0.,.2],[0.,1.,.3],[.5,.5,.4]]),
                  edge_index=torch.tensor([[0,1,2,1],[1,2,0,0]]),edge_attr=torch.ones((4,2))) if graph else None
        bag=(0,2,[1,2],torch.ones((2,3))) if graph else None
        memory.push(np.zeros(4,dtype=np.float32),np.array([i%2],dtype=np.int64),
                    np.ones(4,dtype=np.float32),np.array([i/10],dtype=np.float32),np.array([1.],dtype=np.float32),
                    np.array([True,True]),data,data,bag,bag)


class ReplayShapeTests(unittest.TestCase):
    def test_explicit_feature_axis_and_no_numpy_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            memory=PrioritizedReplayMemoryNumpy(8,4,1,2)
        self.assertFalse([w for w in caught if issubclass(w.category,FutureWarning)])
        self.assertEqual(memory.memory.dtype['action'].shape,(1,))
        self.assertEqual(memory.memory.dtype['reward'].shape,(1,))
        self.assertEqual(memory.memory.dtype['done'].shape,(1,))

    def test_fresh_sample_shapes_and_dtypes(self):
        memory=PrioritizedReplayMemoryNumpy(8,4,1,2)
        fill_memory(memory)
        batch=memory.sample(4)
        self.assertEqual([tuple(t.shape) for t in batch[:6]],[(4,4),(4,1),(4,4),(4,1),(4,1),(4,2)])
        self.assertEqual(batch[1].dtype,torch.int64)
        self.assertEqual(tuple(batch[-1].shape),(4,1))

    def test_restored_scalar_dtype_normalizes_without_discarding_memory(self):
        memory=PrioritizedReplayMemoryNumpy(8,4,1,2)
        fill_memory(memory)
        memory=pickle.loads(pickle.dumps(scalar_legacy_memory(memory)))
        self.assertEqual(memory.memory.dtype['action'].shape,())
        state=(memory.position,memory.full,memory.priorities.copy())
        batch=memory.sample(4)
        self.assertEqual([tuple(t.shape) for t in batch[:6]],[(4,4),(4,1),(4,4),(4,1),(4,1),(4,2)])
        self.assertEqual((memory.position,memory.full),state[:2])
        np.testing.assert_array_equal(memory.priorities,state[2])
        self.assertEqual(len(memory),8)

    def test_batch_size_one_keeps_both_batch_and_feature_axes(self):
        for legacy in (False,True):
            memory=PrioritizedReplayMemoryNumpy(8,4,1,2)
            fill_memory(memory)
            if legacy:scalar_legacy_memory(memory)
            batch=memory.sample(1)
            for field in (1,3,4):self.assertEqual(tuple(batch[field].shape),(1,1))

    def test_structured_state_shape_preserved(self):
        memory=PrioritizedReplayMemoryNumpy(4,(2,2),(1,),2)
        for _ in range(4):
            memory.push(np.ones((2,2)),np.array([0]),np.zeros((2,2)),np.array([1]),np.array([0]),np.ones(2,dtype=bool),None,None,None,None)
        self.assertEqual(tuple(memory.sample(2)[0].shape),(2,2,2))

    def test_full_update_64_samples_fresh_and_legacy_with_scalar_td_targets(self):
        for legacy in (False,True):
            with self.subTest(legacy=legacy):
                torch.manual_seed(42)
                shape=DQNGNNActionShapeConfig(num_nodes=3,max_neighbors=2,node_in=3,edge_in=2,gnn_hidden=8,action_feature_dim=3)
                agent=DQNGNNActionAgentRefactored(shape,batch_size=64,memory_capacity=64,device=torch.device('cpu'))
                fill_memory(agent.memory,64,graph=True)
                if legacy:agent.memory=pickle.loads(pickle.dumps(scalar_legacy_memory(agent.memory)))
                before=next(agent.eval_net.parameters()).detach().clone()
                records={}
                original=agent.memory.sample
                def sample(count):
                    batch=original(count);records['rewards']=batch[3].clone();return batch
                def loss_fn(prediction,target):
                    self.assertEqual(tuple(prediction.shape),(64,1))
                    self.assertEqual(tuple(target.shape),(64,1))
                    torch.testing.assert_close(target,records['rewards'])
                    return (prediction-target).square()
                agent.loss_func=loss_fn
                with patch.object(agent.memory,'sample',side_effect=sample):
                    loss=agent.learn_dqn_neighbor()
                self.assertTrue(math.isfinite(loss))
                self.assertEqual(agent.learn_step_counter,1)
                self.assertFalse(torch.equal(before,next(agent.eval_net.parameters()).detach()))

if __name__=='__main__':unittest.main()
