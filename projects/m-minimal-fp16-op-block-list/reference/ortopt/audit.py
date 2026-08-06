def audit_decoder_graph(graph):
    if not isinstance(graph, dict):
        return False
    return graph.get("has_kv_cache", False) and graph.get("valid_precision", True)
