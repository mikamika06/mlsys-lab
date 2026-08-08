def compute_node_coverage(graph_nodes, partitioned_subgraphs):
    total = len(graph_nodes)
    if total == 0:
        return 0.0
    covered = set()
    for sg in partitioned_subgraphs:
        for node in sg.get("nodes", []):
            if node in graph_nodes:
                covered.add(node)
    return float(len(covered)) / float(total)


def filter_subgraphs(subgraphs, min_nodes=3):
    out = []
    for sg in subgraphs:
        nodes = sg.get("nodes", [])
        if len(nodes) >= min_nodes:
            out.append(sg)
    return out
