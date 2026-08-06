"""Op-graph diff analysis between ipex.optimize and manual channels_last."""


def analyze_layout_conversions(graph):
    """Count explicit layout conversion nodes in a graph."""
    nodes = graph.get("nodes", [])
    count = 0
    for node in nodes:
        op = node.get("op", "")
        if op in ("to", "to_memory_format", "contiguous"):
            target_fmt = node.get("target_format") or node.get("kwargs", {}).get("memory_format")
            if target_fmt == "channels_last" or "channels_last" in str(target_fmt):
                count += 1
    return count


def diff_op_graphs(manual_graph, ipex_graph):
    """Compare manual channels_last graph vs ipex.optimize graph."""
    manual_nodes = manual_graph.get("nodes", [])
    ipex_nodes = ipex_graph.get("nodes", [])

    manual_copies = analyze_layout_conversions(manual_graph)
    ipex_copies = analyze_layout_conversions(ipex_graph)
    redundant_copies_removed = max(0, manual_copies - ipex_copies)

    manual_weights_mem = sum(n.get("weight_bytes", 0) for n in manual_nodes)
    ipex_weights_mem = sum(n.get("weight_bytes", 0) for n in ipex_nodes)

    manual_copies_mem = sum(
        n.get("output_bytes", 0)
        for n in manual_nodes
        if n.get("op") in ("to", "to_memory_format", "contiguous")
    )
    ipex_copies_mem = sum(
        n.get("output_bytes", 0)
        for n in ipex_nodes
        if n.get("op") in ("to", "to_memory_format", "contiguous")
    )

    mem_saved = (manual_weights_mem + manual_copies_mem) - (ipex_weights_mem + ipex_copies_mem)

    return {
        "manual_node_count": len(manual_nodes),
        "ipex_node_count": len(ipex_nodes),
        "redundant_copies_removed": redundant_copies_removed,
        "memory_saved_bytes": max(0, mem_saved),
        "is_ipex_optimized": ipex_copies == 0 and len(ipex_nodes) < len(manual_nodes),
    }
