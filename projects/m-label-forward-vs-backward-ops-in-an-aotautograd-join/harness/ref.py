import random

CHEAP_OPS = {"add", "sub", "mul", "relu", "sigmoid", "copy"}
EXPENSIVE_OPS = {"matmul", "conv2d", "bmm", "einsum", "softmax"}


def generate_synthetic_graphs(seed=42):
    rng = random.Random(seed)
    graphs = []

    for idx in range(10):
        nodes = []
        num_fwd = rng.randint(4, 8)
        num_bwd = rng.randint(4, 8)

        for i in range(num_fwd):
            is_expensive = (i % 2 == 1)
            op = rng.choice(list(EXPENSIVE_OPS if is_expensive else CHEAP_OPS))
            size = rng.randint(100, 500) * 1024
            cost = rng.randint(1000, 5000) if op in EXPENSIVE_OPS else rng.randint(10, 50)
            nodes.append({
                "id": f"node_{i}",
                "op": op,
                "inputs": [f"node_{i-1}"] if i > 0 else ["input_0"],
                "is_fw_target": True,
                "tensor_bytes": size,
                "compute_cost": cost,
                "in_place": False
            })

        for i in range(num_bwd):
            node_idx = num_fwd + i
            is_expensive = (i % 2 == 0)
            op = rng.choice(list(EXPENSIVE_OPS if is_expensive else CHEAP_OPS))
            size = rng.randint(100, 500) * 1024
            cost = rng.randint(1000, 5000) if op in EXPENSIVE_OPS else rng.randint(10, 50)
            nodes.append({
                "id": f"node_{node_idx}",
                "op": f"{op}_backward",
                "inputs": [f"node_{node_idx-1}", f"node_{num_fwd - 1 - (i % num_fwd)}"],
                "is_fw_target": False,
                "tensor_bytes": size,
                "compute_cost": cost,
                "in_place": False
            })

        mut_idx = rng.randint(0, num_fwd - 1)
        nodes[mut_idx]["in_place"] = True
        nodes[mut_idx]["op"] = nodes[mut_idx]["op"] + "_"

        graphs.append({
            "graph_id": f"g_{idx}",
            "nodes": nodes,
            "fwd_count": num_fwd,
            "bwd_count": num_bwd
        })

    return graphs


GRAPHS = generate_synthetic_graphs(42)


def label_joint_graph(graph):
    labeled = []
    for node in graph["nodes"]:
        n = dict(node)
        n["phase"] = "forward" if node["is_fw_target"] else "backward"
        labeled.append(n)
    return labeled


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


def functionalize_graph(graph):
    new_nodes = []
    for node in graph["nodes"]:
        n = dict(node)
        if n.get("in_place", False) or n["op"].endswith("_"):
            base_op = n["op"].rstrip("_")
            n["op"] = base_op
            n["in_place"] = False
            n["out_var"] = f"{n['id']}_out"
        new_nodes.append(n)
    return {"graph_id": graph["graph_id"], "nodes": new_nodes}
