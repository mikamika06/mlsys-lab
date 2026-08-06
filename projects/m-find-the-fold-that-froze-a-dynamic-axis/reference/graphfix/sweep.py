def sweep_dead_nodes_and_orphans(graph):
    nodes = graph.get("nodes", [])
    initializers = graph.get("initializers", {})
    outputs = set(graph.get("outputs", []))

    active_nodes = []
    producer_map = {}
    for node in nodes:
        for out in node.get("outputs", []):
            producer_map[out] = node

    used_tensors = set(outputs)
    queue = list(outputs)
    while queue:
        t = queue.pop(0)
        if t in producer_map:
            n = producer_map[t]
            if n not in active_nodes:
                active_nodes.append(n)
                for inp in n.get("inputs", []):
                    if inp not in used_tensors:
                        used_tensors.add(inp)
                        queue.append(inp)

    active_init = {k: v for k, v in initializers.items() if k in used_tensors}
    new_graph = dict(graph)
    new_graph["nodes"] = active_nodes
    new_graph["initializers"] = active_init
    return new_graph
