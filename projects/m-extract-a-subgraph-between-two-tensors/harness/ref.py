import numpy as np

def make_test_graph():
    nodes = [
        {"name": "n1", "op": "Add", "inputs": ["input_x", "const_a"], "outputs": ["t1"]},
        {"name": "n2", "op": "Mul", "inputs": ["t1", "const_b"], "outputs": ["t2"]},
        {"name": "n3", "op": "Erf", "inputs": ["t2"], "outputs": ["t3"]},
        {"name": "n4", "op": "Add", "inputs": ["t3", "const_c"], "outputs": ["t4"]},
        {"name": "n5", "op": "Mul", "inputs": ["t1", "t4"], "outputs": ["output_y"]},
        {"name": "n6", "op": "Relu", "inputs": ["output_y"], "outputs": ["dead_out"]}
    ]
    initializers = {
        "const_a": np.array([1.0], dtype=np.float32),
        "const_b": np.array([0.70710677], dtype=np.float32),
        "const_c": np.array([1.0], dtype=np.float32),
        "dead_const": np.array([99.0], dtype=np.float32)
    }
    return {"nodes": nodes, "initializers": initializers, "inputs": ["input_x"], "outputs": ["output_y"]}

def extract_subgraph(graph, input_name, output_name):
    nodes = graph["nodes"]
    initializers = graph["initializers"]

    forward_visited = set()
    queue = [output_name]
    while queue:
        curr = queue.pop(0)
        forward_visited.add(curr)
        if curr == input_name:
            continue
        for node in nodes:
            if curr in node["outputs"]:
                for inp in node["inputs"]:
                    if inp not in forward_visited:
                        queue.append(inp)

    backward_visited = set()
    queue = [input_name]
    while queue:
        curr = queue.pop(0)
        backward_visited.add(curr)
        if curr == output_name:
            continue
        for node in nodes:
            if curr in node["inputs"]:
                for out in node["outputs"]:
                    if out not in backward_visited:
                        queue.append(out)

    active_tensors = forward_visited.intersection(backward_visited)
    active_tensors.add(input_name)
    active_tensors.add(output_name)

    kept_nodes = []
    for node in nodes:
        if any(o in active_tensors for o in node["outputs"]) and any(i in active_tensors for i in node["inputs"] or [input_name]):
            kept_nodes.append(node)

    kept_inits = {k: v for k, v in initializers.items() if k in active_tensors}
    return {"nodes": kept_nodes, "initializers": kept_inits, "inputs": [input_name], "outputs": [output_name]}

def fuse_gelu(graph):
    nodes = list(graph["nodes"])
    initializers = dict(graph["initializers"])

    i = 0
    new_nodes = []
    while i < len(nodes):
        if i + 3 < len(nodes):
            n1, n2, n3, n4 = nodes[i], nodes[i+1], nodes[i+2], nodes[i+3]
            if (n1["op"] == "Mul" and n2["op"] == "Erf" and n3["op"] == "Add" and n4["op"] == "Mul" and
                n1["outputs"][0] == n2["inputs"][0] and
                n2["outputs"][0] == n3["inputs"][0] and
                n3["outputs"][0] == n4["inputs"][1] and
                n1["inputs"][0] == n4["inputs"][0]):
                fused = {
                    "name": f"gelu_{n1['name']}",
                    "op": "Gelu",
                    "inputs": [n1["inputs"][0]],
                    "outputs": n4["outputs"],
                    "attributes": {}
                }
                new_nodes.append(fused)
                i += 4
                continue
        new_nodes.append(nodes[i])
        i += 1

    return {"nodes": new_nodes, "initializers": initializers, "inputs": graph["inputs"], "outputs": graph["outputs"]}

def fold_constants(graph):
    nodes = list(graph["nodes"])
    initializers = dict(graph["initializers"])

    folded_inits = dict(initializers)
    new_nodes = []
    for node in nodes:
        if node["op"] == "Add" and len(node["inputs"]) == 2:
            i0, i1 = node["inputs"]
            if i0 in folded_inits and i1 in folded_inits:
                res = folded_inits[i0] + folded_inits[i1]
                out_name = node["outputs"][0]
                folded_inits[out_name] = res
                continue
        new_nodes.append(node)

    active = set(graph["outputs"])
    for n in new_nodes:
        for inp in n["inputs"]:
            active.add(inp)
    for inp in graph["inputs"]:
        active.add(inp)

    cleaned_inits = {k: v for k, v in folded_inits.items() if k in active}
    return {"nodes": new_nodes, "initializers": cleaned_inits, "inputs": graph["inputs"], "outputs": graph["outputs"]}
