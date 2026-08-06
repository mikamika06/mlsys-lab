def compute_node_coverage(graph_nodes, partitioned_subgraphs):
    total = len(graph_nodes)
    if total == 0:
        return 1.0
    supported = set()
    for sub in partitioned_subgraphs:
        for node in sub.get("nodes", []):
            supported.add(node)
    return float(len(supported)) / float(total)


def optimize_subgraph_partitions(graph_nodes, max_op_support):
    subgraphs = []
    current_sub = {"nodes": []}
    for node in graph_nodes:
        if node in max_op_support:
            current_sub["nodes"].append(node)
        else:
            if current_sub["nodes"]:
                subgraphs.append(current_sub)
                current_sub = {"nodes": []}
    if current_sub["nodes"]:
        subgraphs.append(current_sub)
    return subgraphs
