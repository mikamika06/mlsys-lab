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
