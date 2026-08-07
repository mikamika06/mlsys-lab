def fragmentation_cost(graph_nodes, supported_nodes):
    total = len(graph_nodes)
    if total == 0:
        return 0.0
    supported_set = set(supported_nodes)
    unsupported_count = sum(1 for n in graph_nodes if n not in supported_set)
    ratio = unsupported_count / total
    return float(ratio * len(graph_nodes))
