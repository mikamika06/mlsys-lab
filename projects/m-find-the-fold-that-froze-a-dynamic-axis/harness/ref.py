import numpy as np

GRAPHS = [
    {
        "inputs": [{"name": "input_ids", "shape": ["batch_size", "seq_len"]}],
        "nodes": [
            {
                "name": "shape_node_1",
                "op": "Shape",
                "inputs": ["input_ids"],
                "outputs": ["shape_vec"],
                "is_folded": False
            },
            {
                "name": "gather_freeze_node",
                "op": "Gather",
                "inputs": ["shape_vec"],
                "outputs": ["batch_dim"],
                "is_folded": True
            },
            {
                "name": "matmul_active",
                "op": "MatMul",
                "inputs": ["input_ids", "weight_0"],
                "outputs": ["hidden"],
                "is_folded": False
            },
            {
                "name": "orphan_node",
                "op": "Add",
                "inputs": ["hidden", "orphan_bias"],
                "outputs": ["unused_out"],
                "is_folded": False
            }
        ],
        "initializers": {
            "weight_0": list(range(100)),
            "orphan_bias": list(range(10))
        },
        "outputs": ["hidden"]
    },
    {
        "inputs": [
            {"name": "static_input", "shape": [1, 3, 224, 224]},
            {"name": "dynamic_seq", "shape": [1, "seq_len", 512]}
        ],
        "nodes": [
            {
                "name": "fold_static",
                "op": "ConstantFold",
                "inputs": ["static_input"],
                "outputs": ["static_folded"],
                "is_folded": True
            },
            {
                "name": "slice_dynamic_freeze",
                "op": "Slice",
                "inputs": ["dynamic_seq"],
                "outputs": ["frozen_seq_slice"],
                "static_shape": True
            },
            {
                "name": "identity_out",
                "op": "Identity",
                "inputs": ["frozen_seq_slice"],
                "outputs": ["final_output"],
                "is_folded": False
            }
        ],
        "initializers": {
            "init_a": [1.0, 2.0, 3.0]
        },
        "outputs": ["final_output"]
    },
    {
        "inputs": [{"name": "x", "shape": [16, 32]}],
        "nodes": [
            {
                "name": "node_static_fold",
                "op": "ConstantFold",
                "inputs": ["x"],
                "outputs": ["y"],
                "is_folded": True
            }
        ],
        "initializers": {},
        "outputs": ["y"]
    }
]


def find_frozen_dynamic_folds(graph):
    dynamic_tensors = set()
    for inp in graph.get("inputs", []):
        shape = inp.get("shape", [])
        for dim in shape:
            if isinstance(dim, str):
                dynamic_tensors.add(inp["name"])
                break

    tensor_producers = {}
    for node in graph.get("nodes", []):
        for out in node.get("outputs", []):
            tensor_producers[out] = node

    propagated_dynamic = set(dynamic_tensors)
    changed = True
    while changed:
        changed = False
        for node in graph.get("nodes", []):
            if any(inp in propagated_dynamic for inp in node.get("inputs", [])):
                for out in node.get("outputs", []):
                    if out not in propagated_dynamic:
                        propagated_dynamic.add(out)
                        changed = True

    frozen_folds = []
    for node in graph.get("nodes", []):
        if node.get("op") in ("ConstantFold", "Shape", "Gather", "Reshape", "Slice"):
            has_dynamic_input = any(inp in propagated_dynamic for inp in node.get("inputs", []))
            is_constant_output = node.get("is_folded", False) or node.get("static_shape", False)
            if has_dynamic_input and is_constant_output:
                frozen_folds.append(node["name"])

    return sorted(frozen_folds)


def sweep_dead_and_orphans(graph):
    nodes = graph.get("nodes", [])
    initializers = graph.get("initializers", {})
    graph_outputs = set(graph.get("outputs", []))

    producer_map = {}
    for node in nodes:
        for out in node.get("outputs", []):
            producer_map[out] = node

    reachable_nodes = []
    needed_tensors = set(graph_outputs)
    queue = list(graph_outputs)

    while queue:
        tensor = queue.pop(0)
        if tensor in producer_map:
            node = producer_map[tensor]
            if node not in reachable_nodes:
                reachable_nodes.append(node)
                for inp in node.get("inputs", []):
                    if inp not in needed_tensors:
                        needed_tensors.add(inp)
                        queue.append(inp)

    reachable_nodes_ordered = [n for n in nodes if n in reachable_nodes]
    active_initializers = {k: v for k, v in initializers.items() if k in needed_tensors}

    cleaned_graph = dict(graph)
    cleaned_graph["nodes"] = reachable_nodes_ordered
    cleaned_graph["initializers"] = active_initializers
    return cleaned_graph


def compute_simplification_payoff(graph_before, graph_after):
    nodes_before = len(graph_before.get("nodes", []))
    nodes_after = len(graph_after.get("nodes", []))
    init_before = len(graph_before.get("initializers", {}))
    init_after = len(graph_after.get("initializers", {}))

    nodes_removed = nodes_before - nodes_after
    inits_removed = init_before - init_after

    node_reduction_pct = (nodes_removed / nodes_before) * 100.0 if nodes_before > 0 else 0.0
    init_reduction_pct = (inits_removed / init_before) * 100.0 if init_before > 0 else 0.0

    bytes_before = sum(len(v) for v in graph_before.get("initializers", {}).values())
    bytes_after = sum(len(v) for v in graph_after.get("initializers", {}).values())
    bytes_saved = bytes_before - bytes_after

    latency_payoff_estimate_ms = round(nodes_removed * 0.12 + (bytes_saved / 1024.0) * 0.05, 3)

    return {
        "nodes_removed": nodes_removed,
        "initializers_removed": inits_removed,
        "node_reduction_pct": round(node_reduction_pct, 2),
        "init_reduction_pct": round(init_reduction_pct, 2),
        "bytes_saved": bytes_saved,
        "latency_payoff_ms": latency_payoff_estimate_ms
    }
