def detect_fallbacks(graph_spec):
    fallbacks = []
    for i, (op, dtype) in enumerate(graph_spec["nodes"]):
        if "linear" in op and dtype == "float32":
            fallbacks.append(f"node_{i}")
    return fallbacks
