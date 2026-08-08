def count_aten_ops(graph_nodes):
    counts = {}
    for node in graph_nodes:
        if node.get("op") == "call_function":
            target = str(node.get("target", ""))
            if "aten." in target:
                counts[target] = counts.get(target, 0) + 1
    return counts


def identify_decompositions(graph_nodes, target_ops):
    counts = count_aten_ops(graph_nodes)
    found = {op: counts.get(op, 0) for op in target_ops}
    fully_decomposed = all(v == 0 for v in found.values())
    return {"counts": found, "fully_decomposed": fully_decomposed}
