def extract_subgraph(graph, inputs, outputs):
    queue = list(outputs)
    visited = set(inputs)
    for i in inputs:
        visited.add(i)

    kept_nodes = []
    node_map = {n["name"]: n for n in graph["nodes"]}
    output_set = set(outputs)
    input_set = set(inputs)

    active_nodes = []
    for node in graph["nodes"]:
        if any(o in output_set for o in node["outputs"]):
            active_nodes.append(node)

    stack = list(outputs)
    seen_nodes = set()
    while stack:
        out_tensor = stack.pop()
        if out_tensor in input_set:
            continue
        for node in graph["nodes"]:
            if out_tensor in node["outputs"] and node["name"] not in seen_nodes:
                seen_nodes.add(node["name"])
                kept_nodes.append(node)
                for inp in node["inputs"]:
                    if inp not in input_set:
                        stack.append(inp)

    kept_nodes.reverse()
    return {
        "inputs": list(inputs),
        "outputs": list(outputs),
        "nodes": kept_nodes,
        "initializers": {k: v for k, v in graph.get("initializers", {}).items() if any(k in n["inputs"] for n in kept_nodes) or k in inputs}
    }
