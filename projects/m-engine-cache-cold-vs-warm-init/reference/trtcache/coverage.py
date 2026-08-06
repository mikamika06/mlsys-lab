def calculate_node_coverage(subgraphs, ep_nodes):
    if not subgraphs:
        return 0.0
    covered = sum(s.get("node_count", 0) for s in subgraphs if s.get("supported", False))
    total = sum(s.get("node_count", 0) for s in subgraphs)
    if total == 0:
        return 0.0
    return float(covered) / float(total)
