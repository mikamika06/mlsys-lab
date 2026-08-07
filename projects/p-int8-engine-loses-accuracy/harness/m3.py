def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from engine.quantize import place_qdq_nodes

    m = {"qdq_placed": 0.0, "graph_valid": 0.0}
    mock_graph = {"nodes": [{"name": f"layer_{i}", "quantized": True} for i in range(12)]}
    sensitive = [2, 5, 8]

    try:
        new_graph = place_qdq_nodes(mock_graph, sensitive)
    except Exception:
        return m

    if isinstance(new_graph, dict) and "nodes" in new_graph:
        m["qdq_placed"] = 1.0
        qdq_counts = sum(1 for n in new_graph["nodes"] if n.get("has_qdq", False))
        if qdq_counts == 9:
            m["graph_valid"] = 1.0
    return m
