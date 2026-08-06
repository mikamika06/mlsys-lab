def count_aten_ops(exported_program):
    counts = {}
    graph = exported_program.graph
    for node in graph.nodes:
        if node.op == "call_function":
            target_str = str(node.target)
            if "aten." in target_str or "torch.ops.aten." in target_str:
                counts[target_str] = counts.get(target_str, 0) + 1
    return counts


def identify_decompositions(exported_program, target_ops):
    counts = count_aten_ops(exported_program)
    results = {}
    for op in target_ops:
        results[op] = counts.get(op, 0)
    return results
