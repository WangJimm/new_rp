"""Deprecated compatibility facade for the original flat SUMO utility API.

New code should import from ``rp.routing``, ``rp.sumo``, ``rp.graph``, or
``rp.reporting``.  No runtime deprecation warning is emitted because legacy
experiment scripts may treat stdout/stderr as part of their recorded output.
"""

DEPRECATED_FACADE = True
FACADE_REPLACEMENTS = {
    "routing": "rp.routing",
    "sumo": "rp.sumo",
    "graph": "rp.graph",
    "reporting": "rp.reporting",
    "legacy_dijkstra": "legacy.dijkstra_sim",
}

import traci
import torch
from traci import constants as tc
import random
import json

import time

from rp.reporting import inspect_link_api, save_memory_to_txt
from rp.sumo import (
    action_mask_dqn as _action_mask_dqn,
    reached_goal_junction as _reached_goal_junction,
)
from rp.reporting import simple_plot
from rp.graph import freeze_pyg_data

from rp.routing import (
    MAX_SKIP_HOPS,
    MIN_DECISION_LEN,
    VCLASS_FILTER,
    build_macro_candidates,
    edge_allows,
    expand_to_decision_edges,
    is_all_internal_to,
    is_decisionable_edge,
    is_external_edge,
    is_external_node,
    is_internal_edge,
)
from rp.routing import cost_by_length, dijkstra_next_hop, dijkstra_path
from rp.routing.topology import _is_internal_edge
from rp.graph import (
    DECAY,
    FLOOR,
    dynamic_node_feature_update,
    get_edge_index_v2,
)
import rp.graph.features as _graph_features
from rp.routing.geometry import (
    _get_lane_from_id_or_edge,
    _safe_get_lane,
    distance_from_exit_lane_to_target_edge,
    internal_chain_between_edges,
    internal_chains_for_route,
    trace_to_target_edge,
)
import rp.sumo.distance as _traci_distance
import rp.sumo.routing as _traci_routing
import rp.sumo.traffic_signals as _traffic_signals
import rp.sumo.vehicle_spawning as _vehicle_spawning
from legacy.dijkstra_sim import dijkstra_sim as _legacy_dijkstra_sim


def prepare_static(net):
    return _graph_features.prepare_static(net, traci_api=traci)

def collect_dynamic_edge_traveltime(edge_ids, travel_time, MeanSpeed, Occupancy, debug = False):
    return _graph_features.collect_dynamic_edge_traveltime(
        edge_ids,
        travel_time,
        MeanSpeed,
        Occupancy,
        debug,
        traci_api=traci,
    )

def build_data_runtime(static_pack, travel_time, MeanSpeed, Occupancy, debug = False, edge_feature_mode: str = "all"):
    return _graph_features.build_data_runtime(
        static_pack,
        travel_time,
        MeanSpeed,
        Occupancy,
        debug,
        edge_feature_mode,
        traci_api=traci,
    )



def build_dynamic_features(edge_ids):
    return _graph_features.build_dynamic_features(
        edge_ids,
        traci_api=traci,
        traci_constants=tc,
    )

#（可選）node 的 TLS 動態：phase、剩餘時間、排隊等
def tls_node_features(nid):
    raise NotImplementedError(
        "tls_node_features was removed because no feature schema was defined"
    )


def reachable_edges_from(cur_edge):
    return _traci_routing.reachable_edges_from(cur_edge, traci_api=traci)

def candidate_next_junctions(net, cur_edge):
    return _traci_routing.candidate_next_junctions(
        net,
        cur_edge,
        traci_api=traci,
    )

def apply_next_junction(ev_id, chosen_jn, outs):
    return _traci_routing.apply_next_junction(
        ev_id,
        chosen_jn,
        outs,
        traci_api=traci,
    )


""" dijkstra 部分"""

## 預計旅行時間(traci估計)
def cost_by_traveltime(eid):
    return traci.edge.getTraveltime(eid)  # 秒

def apply_full_route(ev, cur_edge, path_edges, suffix_keep=None):
    return _traci_routing.apply_full_route(
        ev,
        cur_edge,
        path_edges,
        suffix_keep=suffix_keep,
        traci_api=traci,
    )

"""
處理決策路段過小，來不及choose action的狀況
"""


def distance_to_next_external_on(net, veh, target_edge_id):
    return _traci_distance.distance_to_next_external_on(
        net,
        veh,
        target_edge_id,
        traci_api=traci,
    )


def remaining_routes_len(net, veh):
    return _traci_distance.remaining_routes_len(
        net,
        veh,
        traci_api=traci,
    )

def remaining_to_exit_external_end(net, veh, internal_edges, exit_lane_id):
    return _traci_distance.remaining_to_exit_external_end(
        net,
        veh,
        internal_edges,
        exit_lane_id,
        traci_api=traci,
    )


GREEN_CHARS = _traffic_signals.GREEN_CHARS

def _pick_active_logic(tls):
    return _traffic_signals._pick_active_logic(tls, traci_api=traci)

def green_ratio_for_link_indices(tls_id: str, link_indices: list[int]):
    return _traffic_signals.green_ratio_for_link_indices(
        tls_id,
        link_indices,
        traci_api=traci,
    )

def lane2edge(lane_id: str) -> str:
    return _traffic_signals.lane2edge(lane_id)

def build_inedge_to_tls_and_indices():
    return _traffic_signals.build_inedge_to_tls_and_indices(traci_api=traci)

def compute_signal_backbone_for_edges(pyg_edge_in_edge_ids, pyg_edge_target_tls_ids):
    return _traffic_signals.compute_signal_backbone_for_edges(
        pyg_edge_in_edge_ids,
        pyg_edge_target_tls_ids,
        traci_api=traci,
    )


def find_tls_and_link_index_for_movement(in_edge_id: str, out_edge_id: str):
    return _traffic_signals.find_tls_and_link_index_for_movement(
        in_edge_id,
        out_edge_id,
        traci_api=traci,
    )

def time_to_green_and_remaining_green(tls, link_idx: int):
    return _traffic_signals.time_to_green_and_remaining_green(
        tls,
        link_idx,
        traci_api=traci,
    )

def spawn_for_rl(veh_id: str, from_edge: str, vtype: str = "ev",
        depart_lane: str = "best", depart_pos: str = "base", depart_speed: str = "max"):
    return _vehicle_spawning.spawn_for_rl(
        veh_id,
        from_edge,
        vtype,
        depart_lane,
        depart_pos,
        depart_speed,
        traci_api=traci,
    )
    

def spawn_evs_testing(base_depart=0, depart_gap=5.0, net = None, adj = None, src_edges = None, goal_nodes = None):
    return _vehicle_spawning.spawn_evs_testing(
        base_depart,
        depart_gap,
        net,
        adj,
        src_edges,
        goal_nodes,
        traci_api=traci,
    )





def spawn_evs(start_edges, cs_loc, N, base_depart=0, depart_gap=5.0, net = None, adj = None, filename = "None", using_env = " ", max_source_attempts=100):
    return _vehicle_spawning.spawn_evs(
        start_edges,
        cs_loc,
        N,
        base_depart,
        depart_gap,
        net,
        adj,
        filename,
        using_env,
        traci_api=traci,
        random_api=random,
        time_api=time,
        pathfinder=dijkstra_path,
        cost_fn=cost_by_traveltime,
        save_routes=save_routes_xml,
        max_source_attempts=max_source_attempts,
    )




def dijkstra_sim(net, static_pack, ev_ids, filesource, rd1 = None, rd2 = None, rd3 = None, rd4 = None, rd5 = None, test = False, set_goal = None, testing_paths = None,
                 traffic_jam = False, using_env = None, ev_route = None, max_simulation_steps=6000):
    return _legacy_dijkstra_sim(
        net,
        static_pack,
        ev_ids,
        filesource,
        rd1,
        rd2,
        rd3,
        rd4,
        rd5,
        test,
        set_goal,
        testing_paths,
        traffic_jam,
        using_env,
        ev_route,
        traci_api=traci,
        torch_api=torch,
        json_api=json,
        count_evs=count_evs_in_rou,
        remaining_distance=remaining_routes_len,
        pathfinder=dijkstra_path,
        cost_fn=cost_by_traveltime,
        apply_route=apply_full_route,
        max_simulation_steps=max_simulation_steps,
    )

def reached_goal_junction(ev, goal_junc, net, eps=1.0, position_tolerance=1e-3):
    return _reached_goal_junction(
        ev,
        goal_junc,
        net,
        eps,
        position_tolerance,
        traci_api=traci,
    )

def action_mask_dqn(net, cur_edge, nid2idx, n=None):
    return _action_mask_dqn(
        net,
        cur_edge,
        nid2idx,
        n,
        traci_api=traci,
        torch_api=torch,
    )

from rp.sumo.route_files import (
    _split_edges,
    batch_keep_first_edge,
    change_route,
    count_evs_in_rou,
    keep_first_edge_in_rou_file,
    save_routes_xml,
)
