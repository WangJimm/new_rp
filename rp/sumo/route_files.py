"""Utilities for reading and rewriting SUMO ``.rou.xml`` files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List


def count_evs_in_rou(rou_path: str, prefix: str = "EV") -> int:
    root = ET.parse(rou_path).getroot()
    count = 0
    for vehicle in root.findall("vehicle"):
        vehicle_id = vehicle.get("id", "")
        if vehicle_id.startswith(prefix):
            count += 1
    return count


def _split_edges(edges_str: str) -> List[str]:
    return [edge for edge in edges_str.strip().split() if edge]


def change_route(
    rou_in: str,
    rou_out: str,
    target_vid: str,
    min_edges_left: int = 1,
) -> bool:
    """Remove the first route edge for one vehicle and save a new file."""
    tree = ET.parse(rou_in)
    root = tree.getroot()

    for vehicle in root.findall("vehicle"):
        if vehicle.get("id") != target_vid:
            continue

        route = vehicle.find("route")
        if route is None:
            return False

        edges_str = (route.get("edges") or "").strip()
        edges: List[str] = [edge for edge in edges_str.split() if edge]
        if len(edges) <= min_edges_left:
            return False

        new_edges = edges[1:]
        if len(new_edges) < min_edges_left:
            return False

        route.set("edges", " ".join(new_edges))
        tree.write(rou_out, encoding="utf-8", xml_declaration=True)
        return True

    return False


def keep_first_edge_in_rou_file(rou_in: str, rou_out: str) -> tuple[int, int]:
    tree = ET.parse(rou_in)
    root = tree.getroot()
    modified = 0
    skipped = 0

    for vehicle in root.findall("vehicle"):
        route = vehicle.find("route")
        if route is None:
            skipped += 1
            continue
        edges_str = (route.get("edges") or "").strip()
        if not edges_str:
            skipped += 1
            continue
        edges = [edge for edge in edges_str.split() if edge]
        if not edges:
            skipped += 1
            continue
        route.set("edges", edges[0])
        modified += 1

    tree.write(rou_out, encoding="utf-8", xml_declaration=True)
    return modified, skipped


def batch_keep_first_edge(
    pattern: str = "*.rou.xml",
    suffix: str = "_firstedge",
    using_env=None,
    general_veh_flows=["free_0900", "norm_1500", "busy_2100", "heavy_2400"],
):
    # ``pattern`` and ``suffix`` are retained for legacy API compatibility.
    for rd3 in range(100, 151):
        for rd4 in range(1, 3):
            for rd5 in range(0, 4):
                ev_routes = str(
                    f"./{using_env}/training_data_v2/"
                    f"ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.rou.xml"
                )
                ev_routes_new = str(
                    f"./{using_env}/new_training_data_2/"
                    f"ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.rou.xml"
                )
                keep_first_edge_in_rou_file(ev_routes, ev_routes_new)

    print("done")


def save_routes_xml(
    ev_routes,
    filename,
    start_depart=13.13,
    depart_gap=5.0,
    vtype_id=None,
    using_env=" ",
):
    """Write generated EV routes using the legacy training-data path schema."""
    output_path = (
        f"./{using_env}/new_training_data_3/"
        f"{filename}_training_set.rou.xml"
    )
    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    for index, edges in enumerate(ev_routes, start=1):
        vehicle_id = f"EV_{index}"
        depart_time = start_depart + (index - 1) * depart_gap
        vehicle_attributes = {
            "id": vehicle_id,
            "depart": f"{depart_time:.2f}",
            "departLane": "best",
            "departSpeed": "max",
        }
        if vtype_id is not None:
            vehicle_attributes["type"] = vtype_id

        vehicle = ET.SubElement(routes, "vehicle", vehicle_attributes)
        if isinstance(edges, (list, tuple)):
            edge_string = " ".join(edges)
        else:
            edge_string = str(edges)
        ET.SubElement(vehicle, "route", {"edges": edge_string})

    tree = ET.ElementTree(routes)
    try:
        ET.indent(tree, space="    ", level=0)
    except AttributeError:
        pass
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


__all__ = [
    "count_evs_in_rou",
    "change_route",
    "keep_first_edge_in_rou_file",
    "batch_keep_first_edge",
    "save_routes_xml",
]
