import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import torch
from torch_geometric.data import Data
import rp.graph.features as current
import rp.experiment.episode_runner as er
from rp.experiment.runtime_state import EMAFeature


def full_collection_runtime(*args, **kwargs):
    """Reference behavior: collect all metrics, then apply the feature mask."""
    collect = current.collect_dynamic_edge_traveltime
    def collect_all(*a, **kw):
        kw["metric_indices"] = (0, 1, 2)
        return collect(*a, **kw)
    with patch.object(current, "collect_dynamic_edge_traveltime", collect_all):
        return current.build_data_runtime(*args, **kwargs)


def pack():
    return dict(data_static=Data(x=torch.tensor([[1.,2.,3.],[0.,1.,2.]]),edge_index=torch.tensor([[0,1],[1,0]]),edge_attr=torch.zeros(2,3)),edge_ids=['a','b'],eid2eidx={'a':0,'b':1},static_edge_attr_norm=torch.tensor([[.1,.2,.3,.4,.5],[.9,.8,.7,.6,.5]]))

def fake_dynamic():
    obj=NS(time=0.,calls={'tt':0,'speed':0,'occ':0})
    def get(kind,edge):
        obj.calls[kind]+=1
        delta=1 if edge=='b' else 0
        return {'tt':20.+obj.time+delta,'speed':10.+obj.time/10+delta,'occ':.2+obj.time/100+delta/10}[kind]
    obj.api=NS(simulation=NS(getTime=lambda:obj.time),edge=NS(getTraveltime=lambda e:get('tt',e),getLastStepMeanSpeed=lambda e:get('speed',e),getLastStepOccupancy=lambda e:get('occ',e)),TraCIException=RuntimeError)
    return obj

def signal_api():
    o=NS(time=10.,phase=0,calls=0)
    o.links={'A':[(('edge1_0','out_0','via'),('edge1_1','out_0','via')),(('edge1_1','other_0','via'),)],'B':[(('edge1_0','out_1','via'),('edge10_0','outbound_0','via'))]}
    def links(t):o.calls+=1;return o.links[t]
    logic=NS(programID='p',getPhases=lambda:[NS(duration=30.,state='Gr'),NS(duration=20.,state='rG')])
    o.api=NS(trafficlight=NS(getIDList=lambda:list(o.links),getControlledLinks=links,getProgram=lambda t:'p',getAllProgramLogics=lambda t:[logic],getPhase=lambda t:o.phase,getNextSwitch=lambda t:30. if o.phase==0 else 50.),simulation=NS(getTime=lambda:o.time,getMinExpectedNumber=lambda:0))
    return o

class OptimizationChecks(unittest.TestCase):
    def test_all_modes_match_full_collection_across_ema_updates(self):
        for mode in sorted(current.VALID_EDGE_FEATURE_MODES):
            with self.subTest(mode=mode):
                f=fake_dynamic();g=fake_dynamic();a=[EMAFeature(half_life_sec=5.) for _ in range(3)];b=[EMAFeature(half_life_sec=5.) for _ in range(3)]
                for now in (0.,1.,5.,5.,30.):
                    f.time=g.time=now
                    x=full_collection_runtime(pack(),*a,edge_feature_mode=mode,traci_api=f.api)
                    y=current.build_data_runtime(pack(),*b,edge_feature_mode=mode,traci_api=g.api)
                    self.assertTrue(torch.equal(x.edge_attr,y.edge_attr),(mode,now));self.assertTrue(torch.equal(x.x,y.x))
    def test_static_occupancy_avoids_unused_getters(self):
        f=fake_dynamic();emas=[EMAFeature(half_life_sec=5.) for _ in range(3)]
        with patch.object(f.api.edge,'getTraveltime',side_effect=AssertionError('unused getter')),patch.object(f.api.edge,'getLastStepMeanSpeed',side_effect=AssertionError('unused getter')):
            x=current.build_data_runtime(pack(),*emas,edge_feature_mode='static_occupancy',traci_api=f.api)
        self.assertEqual(f.calls['occ'],2);self.assertTrue(torch.equal(x.edge_attr[:,:2],torch.zeros(2,2)));self.assertEqual(tuple(x.edge_attr.shape),(2,8))
    def test_debug_keeps_full_metric_collection(self):
        f=fake_dynamic()
        with patch.object(current,'_debug_incident_edges'):
            current.build_data_runtime(pack(),*[EMAFeature(half_life_sec=5.) for _ in range(3)],edge_feature_mode='static_occupancy',debug=True,traci_api=f.api)
        self.assertEqual(f.calls,{'tt':2,'speed':2,'occ':2})
    def test_default_collector_preserves_failure_behavior(self):
        f = fake_dynamic()
        with patch.object(f.api.edge, 'getLastStepMeanSpeed', side_effect=RuntimeError('missing')):
            result = current.collect_dynamic_edge_traveltime(
                ['a'], *[EMAFeature(half_life_sec=5.) for _ in range(3)], traci_api=f.api,
            )
        self.assertTrue(torch.isnan(result).all())
        self.assertEqual(tuple(result.shape), (1, 3))
    def test_topology_matches_legacy_first_match_and_lane_order(self):
        f=signal_api();cache=er.MovementTopologyCache.capture(f.api)
        for pair in [('edge1','out'),('edge1','other'),('edge10','outbound'),('edge1','outbound'),('missing','out')]:
            self.assertEqual(cache.first_tls.get(pair,(None,None)),er.find_tls_and_link_index_for_movement(*pair,traci_api=f.api))
            self.assertEqual(er.resolve_movement_out_lanes(f.api,*pair,cache),er.resolve_movement_out_lanes(f.api,*pair))
        self.assertEqual(cache.outgoing_lanes[('edge1','out')],('out_0','out_1'))
    def test_cached_tls_keeps_live_phases_without_topology_queries(self):
        f=signal_api();cache=er.MovementTopologyCache.capture(f.api);edge=NS(getID=lambda:'out',getToNode=lambda:NS(getID=lambda:'destination'))
        for phase,time in [(0,10.),(1,35.),(0,15.)]:
            f.phase=phase;f.time=time
            expected=er.build_tls_state(f.api,'edge1',['destination'],[edge]);count=f.calls
            actual=er.build_tls_state(f.api,'edge1',['destination'],[edge],cache)
            self.assertEqual(actual,expected);self.assertEqual(f.calls,count)
    def test_runner_rebuilds_cache_on_each_run(self):
        f=signal_api();runner=er.DQNGNNEpisodeRunner(traci=f.api,net=None,static_pack={'nid2idx':{}},agent=NS(memory=[]),travel_time_ema=None,mean_speed_ema=None,occupancy_ema=None)
        run=er.EpisodeRunInput(ev_ids=[],goal_nodes={},evaluate=True,train=False)
        runner.run(run);self.assertIn(('edge1','out'),runner.movement_topology.first_tls)
        f.links={'C':[(('changed_0','newout_0','via'),)]};runner.run(run)
        self.assertNotIn(('edge1','out'),runner.movement_topology.first_tls);self.assertIn(('changed','newout'),runner.movement_topology.first_tls)

if __name__=='__main__':unittest.main(verbosity=2)
