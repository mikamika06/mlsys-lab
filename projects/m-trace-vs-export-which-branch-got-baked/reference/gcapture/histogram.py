def aten_op_histogram(exported_program):
    counts = {}
    graph_module = exported_program.graph_module
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            target = node.target
            name = getattr(target, "__name__", str(target))
            counts[name] = counts.get(name, 0) + 1
    return counts
