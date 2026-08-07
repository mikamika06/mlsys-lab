def partition_graph(graph_module, config):
    ratio = config.get("delegation_ratio", 0.0)
    nodes = graph_module.get("nodes", [])
    delegated_count = int(len(nodes) * ratio)
    delegated = nodes[:delegated_count]
    host = nodes[delegated_count:]
    return {"delegated": delegated, "host": host, "ratio": ratio}
