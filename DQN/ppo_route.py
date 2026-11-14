from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .ppo_torch import TorchPPOAgentRefactored

ROUTE_FEATURE_NAMES = (
    "edge_length_km", "local_travel_time_120s", "distance_to_goal",
    "distance_progress", "goal_dx", "goal_dy", "turn_cos", "turn_sin",
    "next_out_degree", "is_goal", "visit_count", "recent_visit",
)

@dataclass(frozen=True)
class RoutePPOShapeConfig:
    num_nodes: int
    num_edges: int
    max_neighbors: int
    time_normalizer_sec: float = 3600.0

    @property
    def context_dim(self) -> int:
        return self.num_nodes * 2 + 1 + self.num_edges + 4

    @property
    def ppo_input_dim(self) -> int:
        return self.context_dim + self.max_neighbors * len(ROUTE_FEATURE_NAMES)


class RouteActorCritic(nn.Module):
    """Shared candidate scorer; no GNN, shortest-path features or teacher routes."""

    def __init__(self, shape: RoutePPOShapeConfig):
        super().__init__()
        self.shape = shape
        self.context = nn.Sequential(nn.Linear(shape.context_dim, 128), nn.Tanh(),
                                     nn.Linear(128, 128), nn.Tanh())
        self.action_encoder = nn.Sequential(nn.Linear(len(ROUTE_FEATURE_NAMES), 64), nn.Tanh())
        self.scorer = nn.Sequential(nn.Linear(192, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(192, 128), nn.Tanh(), nn.Linear(128, 1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.scorer[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, states, action_masks):
        masks = action_masks.bool().clone()
        empty = ~masks.any(dim=-1)
        masks[empty, 0] = True
        context = self.context(states[:, :self.shape.context_dim])
        candidates = states[:, self.shape.context_dim:].reshape(
            -1, self.shape.max_neighbors, len(ROUTE_FEATURE_NAMES))
        encoded = self.action_encoder(candidates)
        expanded = context.unsqueeze(1).expand(-1, self.shape.max_neighbors, -1)
        logits = self.scorer(torch.cat([expanded, encoded], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~masks, -1e9)
        weights = masks.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)
        values = self.critic(torch.cat([context, pooled], dim=-1)).squeeze(-1)
        return logits, values


class RoutePPOAgent(TorchPPOAgentRefactored):
    """PPO route-state ablation. Keep original ppo_torch_baseline unchanged."""

    uses_route_features = True
    requires_graph_data = False

    def __init__(self, shape: RoutePPOShapeConfig, *, device=None):
        super().__init__(
            shape, lr=3e-4, train_batch_size=1024, sgd_minibatch_size=256,
            num_sgd_iter=10, entropy_coeff=0.01, initial_kl_coeff=0.0,
            target_kl=0.02, network=RouteActorCritic(shape), device=device,
        )

    def configure_route_context(self, net, static_pack, max_decisions):
        self.route_net = net
        self.route_node_index = static_pack["nid2idx"]
        self.route_edge_index = {e: i for i, e in enumerate(static_pack["edge_ids"])}
        self.route_max_decisions = max(1, int(max_decisions))
        xs, ys = zip(*(net.getNode(n).getCoord() for n in self.route_node_index))
        self.route_scale = max(math.hypot(max(xs)-min(xs), max(ys)-min(ys)), 1.0)
        self.route_out_degree = {
            e.getID(): len([n for n in e.getOutgoing()
                            if not n.getID().startswith(":") and n.allows("passenger")])
            for e in net.getEdges() if not e.getID().startswith(":")
        }

    def prepare_route_snapshot(self, *, base_state, traci, cur_edge, goal_node_id,
                               state, next_junctions, outgoing_edges):
        net = self.route_net
        source_id = net.getEdge(cur_edge).getToNode().getID()
        source_idx = self.route_node_index[source_id]
        state.route_visit_counts[source_idx] = state.route_visit_counts.get(source_idx, 0) + 1
        state.route_recent_nodes.append(source_idx)
        source_xy = net.getNode(source_id).getCoord()
        goal_xy = net.getNode(goal_node_id).getCoord()
        current_distance = math.dist(source_xy, goal_xy)
        start_xy = net.getEdge(cur_edge).getFromNode().getCoord()
        incoming = (source_xy[0]-start_xy[0], source_xy[1]-start_xy[1])
        in_norm = max(math.hypot(*incoming), 1e-6)
        kept_nodes, kept_edges, rows = [], [], []
        for node_id in next_junctions:
            candidates = [e for e in outgoing_edges if e.getToNode().getID() == node_id]
            candidates = [e for e in candidates if node_id == goal_node_id
                          or self.route_out_degree.get(e.getID(), 0) > 0]
            if not candidates:
                continue
            # Match apply_next_junction's local minimum-travel-time edge choice.
            values = [(float(traci.edge.getTraveltime(e.getID())), e) for e in candidates]
            travel_time, edge = min(values, key=lambda item: item[0])
            degree = self.route_out_degree.get(edge.getID(), 0)
            xy = net.getNode(node_id).getCoord()
            distance = math.dist(xy, goal_xy)
            outgoing = (xy[0]-source_xy[0], xy[1]-source_xy[1])
            norm = max(math.hypot(*outgoing), 1e-6)
            cosine = (incoming[0]*outgoing[0]+incoming[1]*outgoing[1])/(in_norm*norm)
            sine = (incoming[0]*outgoing[1]-incoming[1]*outgoing[0])/(in_norm*norm)
            idx = self.route_node_index[node_id]
            rows.append([
                min(edge.getLength()/1000.0, 5.0), min(max(travel_time, 0.0)/120.0, 5.0),
                distance/self.route_scale, (current_distance-distance)/self.route_scale,
                (goal_xy[0]-xy[0])/self.route_scale, (goal_xy[1]-xy[1])/self.route_scale,
                max(-1.0, min(1.0, cosine)), max(-1.0, min(1.0, sine)),
                min(degree/self.a_size, 1.0), float(node_id == goal_node_id),
                min(state.route_visit_counts.get(idx, 0), 5)/5.0,
                float(idx in state.route_recent_nodes),
            ])
            kept_nodes.append(node_id)
            kept_edges.append(edge)
        incoming_one_hot = torch.zeros(self.shape.num_edges)
        if cur_edge in self.route_edge_index:
            incoming_one_hot[self.route_edge_index[cur_edge]] = 1
        global_features = torch.tensor([
            max(0.0, 1.0-state.decisions/self.route_max_decisions),
            current_distance/self.route_scale,
            len(state.route_visit_counts)/self.num_nodes,
            min(state.route_visit_counts[source_idx], 5)/5.0,
        ])
        padded = torch.zeros(self.a_size, len(ROUTE_FEATURE_NAMES))
        features = torch.tensor(rows, dtype=torch.float32).reshape(-1, len(ROUTE_FEATURE_NAMES))
        padded[:len(rows)] = features
        full = torch.cat([base_state.detach().cpu(), incoming_one_hot, global_features, padded.flatten()])
        return full.to(self.device), kept_nodes, kept_edges, features

    def concat_state_action_features(self, state, action_features=None):
        """Terminal-state adapter; no candidate observation exists after termination."""
        if state.numel() == self.shape.ppo_input_dim:
            return state
        padding = self.shape.ppo_input_dim - state.numel()
        if padding < 0:
            raise ValueError("Route observation is larger than its configured shape")
        return torch.cat([state, torch.zeros(padding, device=state.device)])

    def load_training_state(self, state):
        schema = state.get("route_feature_schema")
        if schema != list(ROUTE_FEATURE_NAMES):
            raise ValueError("Checkpoint is not compatible with the PPO route feature schema; start a fresh run")
        super().load_training_state(state)

    def save_tmp_file(self, ep, filename="tmp_file.pt"):
        torch.save({"episode": ep, "shape": self.shape, "agent_type": "ppo_route",
                    "agent_state": self.get_training_state()}, filename)

    def get_training_state(self):
        state = super().get_training_state()
        state["route_feature_schema"] = list(ROUTE_FEATURE_NAMES)
        state["route_training"] = {"lr": 3e-4, "rollout": 1024, "minibatch": 256,
                                   "epochs": 10, "target_kl": 0.02}
        return state
