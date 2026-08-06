def fuse_graph(graph_def):
    nodes = list(graph_def.get("nodes", []))
    fused = []
    skip = False
    for i, node in enumerate(nodes):
        if skip:
            skip = False
            continue
        if node.get("op") == "MatMul" and i + 1 < len(nodes) and nodes[i + 1].get("op") == "Add":
            fused.append({"op": "FusedAttention", "inputs": node.get("inputs"), "outputs": nodes[i + 1].get("outputs")})
            skip = True
        else:
            fused.append(node)
    return {"nodes": fused, "name": graph_def.get("name", "fused")}
