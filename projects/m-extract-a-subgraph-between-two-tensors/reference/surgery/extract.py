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
