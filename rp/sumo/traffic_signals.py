"""Traffic-signal feature helpers with explicit TraCI injection."""

from __future__ import annotations

from collections import defaultdict


GREEN_CHARS = set("gG")


def _pick_active_logic(tls_id, *, traci_api):
    """Select the active program logic, falling back to the first logic."""
    program = traci_api.trafficlight.getProgram(tls_id)
    logics = traci_api.trafficlight.getAllProgramLogics(tls_id)
    for logic in logics:
        try:
            if getattr(logic, "programID", None) == program:
                return logic
        except Exception:
            pass
    return logics[0] if logics else None


def green_ratio_for_link_indices(tls_id: str, link_indices: list[int], *, traci_api):
    """Return cycle duration and mean green ratio for controlled links."""
    logic = _pick_active_logic(tls_id, traci_api=traci_api)
    phases = logic.getPhases()
    cycle = sum(float(phase.duration) for phase in phases)
    if cycle == 0:
        return 0.0, 0.0

    ratios = []
    for link_index in link_indices:
        green_duration = 0.0
        for phase in phases:
            if phase.state[link_index] in GREEN_CHARS:
                green_duration += float(phase.duration)
        ratios.append(green_duration / cycle)
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return cycle, mean_ratio


def lane2edge(lane_id: str) -> str:
    """Remove the final SUMO lane-index suffix from a lane ID."""
    return lane_id.rsplit("_", 1)[0]


def build_inedge_to_tls_and_indices(*, traci_api):
    """Map incoming edge IDs to TLS IDs and controlled-link indices."""
    table = defaultdict(lambda: defaultdict(list))
    for tls_id in traci_api.trafficlight.getIDList():
        links = traci_api.trafficlight.getControlledLinks(tls_id)
        for link_index, group in enumerate(links):
            for in_lane, _out_lane, _via_lane in group:
                in_edge = lane2edge(in_lane)
                table[in_edge][tls_id].append(link_index)
    return table


def compute_signal_backbone_for_edges(
    pyg_edge_in_edge_ids,
    pyg_edge_target_tls_ids,
    *,
    traci_api,
):
    """Compute cycle length and green-ratio features for PyG edges."""
    edge_count = len(pyg_edge_in_edge_ids)
    inedge_to_tls = build_inedge_to_tls_and_indices(traci_api=traci_api)

    cycle_lengths = [0.0] * edge_count
    green_ratios = [1.0] * edge_count
    for index in range(edge_count):
        in_edge = pyg_edge_in_edge_ids[index]
        tls_id = pyg_edge_target_tls_ids[index]
        if not tls_id:
            continue
        link_indices = inedge_to_tls.get(in_edge, {}).get(tls_id, [])
        if not link_indices:
            continue
        cycle, ratio = green_ratio_for_link_indices(
            tls_id,
            link_indices,
            traci_api=traci_api,
        )
        cycle_lengths[index] = cycle
        green_ratios[index] = ratio
    return cycle_lengths, green_ratios


def find_tls_and_link_index_for_movement(
    in_edge_id: str,
    out_edge_id: str,
    *,
    traci_api,
):
    """Return the first TLS and link index matching an edge movement."""
    for tls_id in traci_api.trafficlight.getIDList():
        links = traci_api.trafficlight.getControlledLinks(tls_id)
        for link_index, link_group in enumerate(links):
            for in_lane, out_lane, _via_lane in link_group:
                if (
                    lane2edge(in_lane) == in_edge_id
                    and lane2edge(out_lane) == out_edge_id
                ):
                    return tls_id, link_index
    return None, None


def time_to_green_and_remaining_green(tls_id, link_idx: int, *, traci_api):
    """Estimate time to the next green and current/next green duration."""
    now = traci_api.simulation.getTime()
    current_index = traci_api.trafficlight.getPhase(tls_id)
    next_switch = traci_api.trafficlight.getNextSwitch(tls_id)
    current_remainder = max(0.0, next_switch - now)
    logic = _pick_active_logic(tls_id, traci_api=traci_api)
    phases = logic.getPhases()

    current_character = phases[current_index].state[link_idx]
    if current_character in GREEN_CHARS:
        return 0.0, current_remainder

    wait = current_remainder
    phase_index = (current_index + 1) % len(phases)
    visited = 0
    while visited < len(phases):
        character = phases[phase_index].state[link_idx]
        if character in GREEN_CHARS:
            return wait, float(phases[phase_index].duration)
        wait += float(phases[phase_index].duration)
        phase_index = (phase_index + 1) % len(phases)
        visited += 1
    return float("inf"), 0.0


__all__ = [
    "GREEN_CHARS",
    "green_ratio_for_link_indices",
    "lane2edge",
    "build_inedge_to_tls_and_indices",
    "compute_signal_backbone_for_edges",
    "find_tls_and_link_index_for_movement",
    "time_to_green_and_remaining_green",
]
