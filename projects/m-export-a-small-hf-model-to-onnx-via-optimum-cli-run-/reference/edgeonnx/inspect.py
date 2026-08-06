def get_node_placement(graph_spec):
    placement = {}
    for node in graph_spec.get("nodes", []):
        placement[node["name"]] = node.get("provider", "CPUExecutionProvider")
    return placement


def find_fallback_nodes(graph_spec, provider="CoreMLExecutionProvider"):
    fallbacks = []
    for node in graph_spec.get("nodes", []):
        if node.get("provider") != provider:
            fallbacks.append(node["name"])
    return fallbacks
