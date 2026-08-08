def sanitize_qdq_nodes(graph_nodes, invalid_qdq_patterns, sensitive_layers):
    sanitized = {}
    invalid_set = set(invalid_qdq_patterns)
    for layer_name, nodes in graph_nodes.items():
        cleaned_nodes = []
        for n in nodes:
            if n in invalid_set:
                continue
            cleaned_nodes.append(n)
        sanitized[layer_name] = cleaned_nodes
    return sanitized
