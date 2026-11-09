"""Static and runtime PyTorch Geometric feature construction."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data

from rp.routing import is_external_edge, is_external_node
from rp.sumo import compute_signal_backbone_for_edges, build_inedge_to_tls_and_indices


DECAY = 0.95
FLOOR = 0

VALID_EDGE_FEATURE_MODES = {
    "all",
    "static_only",
    "dynamic_only",
    "travel_time_only",
    "avg_speed_only",
    "occupancy_only",
    "static_travel_time",
    "static_occupancy",
    "static_travel_time_occupancy",
    "none",
}


def dynamic_node_feature_update(
    data_static,
    target,
    recent_node,
    ever_visit,
    visit_age,
):
    """Append target, visit, recent, and visit-age columns to node features."""
    device = data_static.x.device
    dtype = data_static.x.dtype
    ever_visit = ever_visit.to(device=device, dtype=dtype)
    visit_age = visit_age.to(device=device, dtype=dtype)

    is_target = torch.zeros(len(data_static.x), device=device, dtype=dtype)
    is_target[target] = 1
    is_target = is_target.unsqueeze(1)

    recent_nodes = list(recent_node)
    if recent_nodes:
        ever_visit[recent_nodes[-1]] = 1

    recent_flag = torch.zeros(
        len(data_static.x),
        1,
        device=device,
        dtype=dtype,
    )
    for node_id in recent_nodes:
        recent_flag[node_id] = 1.0

    visit_age[:] = torch.clamp(DECAY * visit_age, min=FLOOR)
    if recent_nodes:
        visit_age[recent_nodes[-1]] = 1.0

    new_x = torch.cat(
        (data_static.x, is_target, ever_visit, recent_flag, visit_age),
        dim=1,
    )
    new_data_static = Data(
        x=new_x,
        edge_index=data_static.edge_index,
        edge_attr=data_static.edge_attr,
    )
    return new_data_static, ever_visit, visit_age


def get_edge_index_v2(net):
    """Build the legacy junction graph and its static three-column schema."""
    nodes = [node for node in net.getNodes() if is_external_node(node)]
    node_index = {node.getID(): index for index, node in enumerate(nodes)}
    external_edges = [
        edge for edge in net.getEdges() if is_external_edge(edge)
    ]
    graph_edges = [
        edge
        for edge in external_edges
        if edge.getFromNode().getID() in node_index
        and edge.getToNode().getID() in node_index
    ]

    node_features = np.zeros((len(nodes), 3), np.float32)
    for node in nodes:
        index = node_index[node.getID()]
        signalized = (
            1.0 if node.getType().startswith("traffic_light") else 0.0
        )
        in_degree = len(
            [
                edge
                for edge in node.getIncoming()
                if edge.getFunction() != "internal"
            ]
        )
        out_degree = len(
            [
                edge
                for edge in node.getOutgoing()
                if edge.getFunction() != "internal"
            ]
        )
        node_features[index] = [signalized, in_degree, out_degree]

    sources, targets, edge_features = [], [], []
    for edge in graph_edges:
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        sources.append(node_index[from_node])
        targets.append(node_index[to_node])
        edge_features.append(
            [edge.getLength(), edge.getSpeed(), len(edge.getLanes())]
        )

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    data_static = Data(
        x=torch.from_numpy(node_features),
        edge_index=edge_index,
        edge_attr=torch.tensor(
            np.asarray(edge_features, dtype=np.float32).reshape(-1, 3),
            dtype=torch.float32,
        ),
    )
    edge_ids = [edge.getID() for edge in graph_edges]

    adjacency = defaultdict(list)
    edge_lengths = {}
    for edge in graph_edges:
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        edge_id = edge.getID()
        edge_lengths[edge_id] = edge.getLength()
        adjacency[from_node].append((to_node, edge_id))

    return data_static, edge_ids, node_index, adjacency, edge_lengths


def _minmax(tensor):
    minimum, maximum = tensor.min(), tensor.max()
    return (
        (tensor - minimum) / (maximum - minimum + 1e-8),
        (minimum, maximum),
    )


def prepare_static(net, *, traci_api, prefer_controlled_tls=False):
    """Build reusable static graph data and normalized edge attributes."""
    data_static, edge_ids, node_index, adjacency, edge_lengths = (
        get_edge_index_v2(net)
    )
    edge_to_index = {edge_id: index for index, edge_id in enumerate(edge_ids)}
    static_edge_attr = data_static.edge_attr.clone()

    tls_pool = traci_api.trafficlight.getIDList()
    controlled = build_inedge_to_tls_and_indices(traci_api=traci_api) if prefer_controlled_tls else {}
    target_tls_ids = []
    for edge_id in edge_ids:
        target_node = net.getEdge(edge_id).getToNode().getID()
        controllers = controlled.get(edge_id, {})
        if controllers:
            # SUMO-generated GS_* controller IDs need not equal junction IDs.
            tls_id = target_node if target_node in controllers else sorted(controllers)[0]
            target_tls_ids.append(tls_id)
        else:
            target_tls_ids.append(target_node if target_node in tls_pool else None)
    cycle_lengths, green_ratios = compute_signal_backbone_for_edges(
        edge_ids,
        target_tls_ids,
        traci_api=traci_api,
    )
    cycle_lengths = torch.tensor(cycle_lengths, dtype=torch.float32)
    cycle_lengths /= 60
    green_ratios = torch.tensor(green_ratios, dtype=torch.float32)

    normalized_length, length_range = _minmax(static_edge_attr[:, 0])
    normalized_speed, speed_range = _minmax(static_edge_attr[:, 1])
    normalized_lanes, lane_range = _minmax(static_edge_attr[:, 2])
    static_edge_attr_norm = torch.stack(
        [
            normalized_length,
            normalized_speed,
            normalized_lanes,
            cycle_lengths,
            green_ratios,
        ],
        dim=1,
    )

    return {
        "data_static": data_static,
        "edge_ids": edge_ids,
        "eid2eidx": edge_to_index,
        "nid2idx": node_index,
        "adj": adjacency,
        "edge_len": edge_lengths,
        "static_edge_attr_norm": static_edge_attr_norm,
        "edge_mm": {
            "length": length_range,
            "speed": speed_range,
            "lanes": lane_range,
        },
    }


def collect_dynamic_edge_traveltime(
    edge_ids,
    travel_time,
    mean_speed,
    occupancy,
    debug=False,
    *,
    traci_api,
    metric_indices=(0, 1, 2),
):
    """Collect selected dynamic columns; unrequested columns stay zero.

    The default retains the complete legacy collection API. Runtime feature
    modes are fixed for an agent, so unused EMA histories need not be updated.
    """
    current_time = traci_api.simulation.getTime()
    requested = set(metric_indices)
    collect_time, collect_speed, collect_occupancy = (i in requested for i in range(3))
    values = []
    for edge_id in edge_ids:
        travel_time_value = mean_speed_value = occupancy_value = 0.0
        try:
            if collect_time:
                travel_time_value = travel_time.update(
                    edge_id, traci_api.edge.getTraveltime(edge_id), current_time,
                )
            if collect_speed:
                mean_speed_value = mean_speed.update(
                    edge_id, traci_api.edge.getLastStepMeanSpeed(edge_id), current_time,
                )
            if collect_occupancy:
                occupancy_value = occupancy.update(
                    edge_id, traci_api.edge.getLastStepOccupancy(edge_id), current_time,
                )
        except traci_api.TraCIException:
            travel_time_value = mean_speed_value = occupancy_value = np.nan
        values.append([travel_time_value, mean_speed_value, occupancy_value])
    return torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(-1, 3))


def _debug_incident_edges(
    static_pack,
    dynamic_values,
    normalized_values,
    edges_to_check,
    tag="",
):
    edge_to_index = static_pack.get("edge2idx") or static_pack.get("eid2eidx")
    if edge_to_index is None:
        print(f"\n[DEBUG {tag}] edge index mapping not found")
        return
    print(f"\n[DEBUG {tag}] incident edge features")
    for edge_id in edges_to_check:
        if edge_id not in edge_to_index:
            print(f"  {edge_id}: NOT IN edge_ids")
            continue
        index = edge_to_index[edge_id]
        raw_travel_time, raw_average, raw_occupancy = dynamic_values[index].tolist()
        norm_travel_time, norm_average, norm_occupancy = normalized_values[index].tolist()
        print(
            f"  {edge_id} idx={index:5d} "
            f"raw(tt={raw_travel_time:.3f}, avg={raw_average:.3f}, "
            f"occ={raw_occupancy:.3f}) | "
            f"norm(tt={norm_travel_time:.3f}, avg={norm_average:.3f}, "
            f"occ={norm_occupancy:.3f})"
        )


def build_data_runtime(
    static_pack,
    travel_time,
    mean_speed,
    occupancy,
    debug=False,
    edge_feature_mode: str = "all",
    *,
    traci_api,
):
    """Build runtime graph data with the fixed eight-column edge schema."""
    data_static = static_pack["data_static"]
    edge_ids = static_pack["edge_ids"]
    static_edge_attr_norm = static_pack["static_edge_attr_norm"]

    if edge_feature_mode not in VALID_EDGE_FEATURE_MODES:
        raise ValueError(f"Unknown edge_feature_mode: {edge_feature_mode}")

    dynamic_values = collect_dynamic_edge_traveltime(
        edge_ids,
        travel_time,
        mean_speed,
        occupancy,
        debug,
        traci_api=traci_api,
        metric_indices=(0, 1, 2) if debug else {
            "all": (0, 1, 2),
            "dynamic_only": (0, 1, 2),
            "static_only": (),
            "none": (),
            "travel_time_only": (0,),
            "static_travel_time": (0,),
            "avg_speed_only": (1,),
            "occupancy_only": (2,),
            "static_occupancy": (2,),
            "static_travel_time_occupancy": (0, 2),
        }[edge_feature_mode],
    )
    dynamic_values = torch.nan_to_num(
        dynamic_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    max_travel_time = 600.0
    max_speed = 30.0
    max_log_travel_time = torch.log1p(
        torch.tensor(
            max_travel_time,
            dtype=dynamic_values.dtype,
            device=dynamic_values.device,
        )
    )
    travel_time_raw = torch.clamp(
        dynamic_values[:, 0],
        min=0.0,
        max=max_travel_time,
    )
    average_speed_raw = torch.clamp(
        dynamic_values[:, 1],
        min=0.0,
        max=max_speed,
    )
    occupancy_raw = torch.clamp(
        dynamic_values[:, 2],
        min=0.0,
        max=1.0,
    )

    normalized_dynamic = torch.stack(
        [
            torch.log1p(travel_time_raw) / max_log_travel_time,
            average_speed_raw / max_speed,
            occupancy_raw,
        ],
        dim=1,
    )
    normalized_dynamic = torch.nan_to_num(
        normalized_dynamic,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    normalized_dynamic = torch.clamp(
        normalized_dynamic,
        min=0.0,
        max=1.0,
    )
    static_edge_attr_norm = torch.nan_to_num(
        static_edge_attr_norm,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    static_edge_attr_norm = torch.clamp(
        static_edge_attr_norm,
        min=0.0,
        max=1.0,
    )

    if debug:
        _debug_incident_edges(
            static_pack,
            dynamic_values,
            normalized_dynamic,
            edges_to_check=["5707608#2", "5707608#0"],
            tag="after_collect",
        )

    if edge_feature_mode == "all":
        dynamic_part = normalized_dynamic
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "static_only":
        dynamic_part = torch.zeros_like(normalized_dynamic)
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "dynamic_only":
        dynamic_part = normalized_dynamic
        static_part = torch.zeros_like(static_edge_attr_norm)
    elif edge_feature_mode == "travel_time_only":
        dynamic_part = torch.zeros_like(normalized_dynamic)
        dynamic_part[:, 0] = normalized_dynamic[:, 0]
        static_part = torch.zeros_like(static_edge_attr_norm)
    elif edge_feature_mode == "avg_speed_only":
        dynamic_part = torch.zeros_like(normalized_dynamic)
        dynamic_part[:, 1] = normalized_dynamic[:, 1]
        static_part = torch.zeros_like(static_edge_attr_norm)
    elif edge_feature_mode == "static_travel_time":
        dynamic_part = torch.zeros_like(normalized_dynamic)
        dynamic_part[:, 0] = normalized_dynamic[:, 0]
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "static_occupancy":
        dynamic_part = torch.zeros_like(normalized_dynamic)
        dynamic_part[:, 2] = normalized_dynamic[:, 2]
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "static_travel_time_occupancy":
        dynamic_part = torch.zeros_like(normalized_dynamic)
        dynamic_part[:, 0] = normalized_dynamic[:, 0]
        dynamic_part[:, 2] = normalized_dynamic[:, 2]
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "none":
        dynamic_part = torch.zeros_like(normalized_dynamic)
        static_part = torch.zeros_like(static_edge_attr_norm)
    else:
        dynamic_part = torch.zeros_like(normalized_dynamic)
        dynamic_part[:, 2] = normalized_dynamic[:, 2]
        static_part = torch.zeros_like(static_edge_attr_norm)

    edge_attr = torch.cat([dynamic_part, static_part], dim=1)
    edge_attr = torch.nan_to_num(
        edge_attr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)

    node_features = data_static.x.clone()
    node_features = torch.nan_to_num(
        node_features,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return Data(
        x=node_features,
        edge_index=data_static.edge_index,
        edge_attr=edge_attr,
    )


def build_dynamic_features(edge_ids, *, traci_api, traci_constants):
    """Build subscription-based dynamic edge features."""
    subscriptions = traci_api.edge.getAllSubscriptionResults()
    dynamic = []
    for edge_id in edge_ids:
        result = subscriptions.get(edge_id, {})
        dynamic.append(
            [
                result.get(traci_constants.LAST_STEP_MEAN_SPEED, 0.0),
                result.get(traci_constants.LAST_STEP_OCCUPANCY, 0.0),
                result.get(traci_constants.LAST_STEP_VEHICLE_NUMBER, 0.0),
            ]
        )
    return torch.tensor(np.array(dynamic), dtype=torch.float32)


def tls_node_features(_node_id):
    """Fail explicitly: the former placeholder had no defined schema."""
    raise NotImplementedError(
        "tls_node_features was removed because no feature schema was defined"
    )


__all__ = [
    "DECAY",
    "FLOOR",
    "VALID_EDGE_FEATURE_MODES",
    "dynamic_node_feature_update",
    "get_edge_index_v2",
    "prepare_static",
    "collect_dynamic_edge_traveltime",
    "build_data_runtime",
    "build_dynamic_features",
]
