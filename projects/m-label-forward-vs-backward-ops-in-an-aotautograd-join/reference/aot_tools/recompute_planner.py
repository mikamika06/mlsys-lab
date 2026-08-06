from aot_tools.graph_labeler import label_joint_graph


def evaluate_recompute_tradeoff(graph, max_recompute_cost=100):
    labeled = label_joint_graph(graph)
    total_saved_bytes = 0
    decisions = {}

    for node in labeled:
        if node["phase"] == "forward":
            if node["compute_cost"] <= max_recompute_cost:
                decisions[node["id"]] = "recompute"
                total_saved_bytes += node["tensor_bytes"]
            else:
                decisions[node["id"]] = "save"

    return {"decisions": decisions, "saved_bytes": total_saved_bytes}
