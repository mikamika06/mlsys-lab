def classify_support(graph):
    return [op.get("supported", False) for op in graph.get("ops", [])]
