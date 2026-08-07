from pt2e_counts.nodes import extract_node_stats, classify_node


def analyze_graph_counts(graph):
    return extract_node_stats(graph)


def compute_conversion_deltas(orig_graph, prep_graph, conv_graph):
    orig_stats = extract_node_stats(orig_graph)
    prep_stats = extract_node_stats(prep_graph)
    conv_stats = extract_node_stats(conv_graph)
    prep_delta = {k: prep_stats.get(k, 0) - orig_stats.get(k, 0) for k in set(list(orig_stats.keys()) + list(prep_stats.keys()))}
    conv_delta = {k: conv_stats.get(k, 0) - prep_stats.get(k, 0) for k in set(list(prep_stats.keys()) + list(conv_stats.keys()))}
    return {
        "orig": orig_stats,
        "prep": prep_stats,
        "conv": conv_stats,
        "prep_delta": prep_delta,
        "conv_delta": conv_delta
    }


def check_node_invariant(graph):
    stats = extract_node_stats(graph)
    return stats.get("total", 0) >= sum(v for k, v in stats.items() if k != "total")
