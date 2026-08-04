"""AlphaProof search-tree transformations and difficulty scoring."""

import argparse
import json
from pathlib import Path
from typing import Any


def hardest_subproblem_tree(
    search_tree: dict[str, Any],
) -> dict[str, Any]:
    """Replace every AND node with its lowest-value OR child."""

    source_nodes = {
        int(node["id"]): node for node in search_tree["nodes"]
    }
    projected_nodes: list[dict[str, Any]] = []

    def project_node(node_id: int) -> int:
        node = source_nodes[node_id]
        if node["node_type"] != "OR":
            raise ValueError("The projected node must be an OR node.")

        projected_id = len(projected_nodes)
        projected_nodes.append({})
        children = []
        for edge in node["children"]:
            child = source_nodes[int(edge["node_id"])]
            projected_edge = {"action": edge["action"]}
            if child["node_type"] == "AND":
                if not child["children"]:
                    raise ValueError("An AND node must have children.")
                selected_edge = min(
                    child["children"],
                    key=lambda item: source_nodes[int(item["node_id"])][
                        "value"
                    ],
                )
                selected_id = int(selected_edge["node_id"])
                selected = source_nodes[selected_id]
                if selected["node_type"] != "OR":
                    raise ValueError("An AND child must be an OR node.")
                projected_edge["collapsed_and_node_id"] = int(child["id"])
                projected_edge["selected_focus_action"] = selected_edge[
                    "action"
                ]
                projected_edge["node_id"] = project_node(selected_id)
            else:
                projected_edge["node_id"] = project_node(int(child["id"]))
            children.append(projected_edge)

        projected_nodes[projected_id] = {
            **{key: value for key, value in node.items() if key != "children"},
            "id": projected_id,
            "children": children,
        }
        return projected_id

    root_id = project_node(int(search_tree["root_id"]))
    return {"root_id": root_id, "nodes": projected_nodes}


def hardest_subproblem_solve_rate(
    search_tree: dict[str, Any],
) -> dict[str, int | float]:
    """Calculate the proven fraction of unique leaf-parent OR nodes."""

    projected = hardest_subproblem_tree(search_tree)
    nodes = {int(node["id"]): node for node in projected["nodes"]}
    leaf_ids = {
        node_id for node_id, node in nodes.items() if not node["children"]
    }
    frontier_ids = {
        node_id
        for node_id, node in nodes.items()
        if any(
            int(edge["node_id"]) in leaf_ids
            for edge in node["children"]
        )
    }
    if not frontier_ids:
        frontier_ids = {int(projected["root_id"])}

    solved = sum(bool(nodes[node_id]["proven"]) for node_id in frontier_ids)
    total = len(frontier_ids)
    return {
        "solved_frontier_nodes": solved,
        "total_frontier_nodes": total,
        "solve_rate": solved / total,
    }


def main() -> None:
    """Calculate the metric from one AlphaProof result JSON object."""

    parser = argparse.ArgumentParser()
    parser.add_argument("search_tree", type=Path)
    parser.add_argument("--projected-tree-output", type=Path)
    args = parser.parse_args()

    with args.search_tree.open(encoding="utf-8") as file:
        search_tree = json.load(file)["tree"]
    if args.projected_tree_output is not None:
        projected = hardest_subproblem_tree(search_tree)
        with args.projected_tree_output.open("w", encoding="utf-8") as file:
            json.dump(projected, file)
    print(json.dumps(hardest_subproblem_solve_rate(search_tree)))


if __name__ == "__main__":
    main()
