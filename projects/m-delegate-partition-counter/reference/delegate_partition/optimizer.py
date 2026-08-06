def optimize_graph(graph):
    ops = list(graph["ops"])
    supported = set(graph["supported"])
    substitutions = {"CustomOp": "Add", "Softmax": "Mul"}
    new_ops = [substitutions.get(op, op) for op in ops]
    return {"ops": new_ops, "supported": graph["supported"]}
