def analyze_propagation(graph, shape_map):
    results = []
    for node in graph["nodes"]:
        resolved = [shape_map.get(dim, dim) for dim in node["shape"]]
        results.append({"op": node["op"], "resolved_shape": resolved})
    return results


def compile_growth_factor(enumerated_list):
    return sum(len(s) for s in enumerated_list) * 1.5


def explain_rejection(graph, target_shape):
    for dim in graph.get("symbolics", {}):
        if dim not in target_shape:
            return f"unresolved dim {dim}"
    return "valid"
