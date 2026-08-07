def count_prepare_nodes(graph):
    counts = {"observer": 0, "q": 0, "dq": 0, "base": 0}
    for node in graph.get("nodes", []):
        target = str(node.get("target", ""))
        op = str(node.get("op", ""))
        if "observer" in target or op == "observer":
            counts["observer"] += 1
        elif "quantize" in target or "q_node" in target:
            counts["q"] += 1
        elif "dequantize" in target or "dq_node" in target:
            counts["dq"] += 1
        else:
            counts["base"] += 1
    return counts


def count_convert_nodes(graph):
    counts = {"quantized_op": 0, "base": 0}
    for node in graph.get("nodes", []):
        target = str(node.get("target", ""))
        if "quantized" in target or "qlinear" in target or "q_op" in target:
            counts["quantized_op"] += 1
        else:
            counts["base"] += 1
    return counts


def compute_reduction_ratio(prep, conv):
    total_prep = sum(prep.values())
    total_conv = sum(conv.values())
    if total_prep == 0:
        return 0.0
    return float(total_conv) / float(total_prep)
