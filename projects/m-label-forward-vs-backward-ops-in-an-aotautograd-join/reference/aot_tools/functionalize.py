def functionalize_graph(graph):
    new_nodes = []
    for node in graph["nodes"]:
        n = dict(node)
        if n.get("in_place", False) or n["op"].endswith("_"):
            base_op = n["op"].rstrip("_")
            n["op"] = base_op
            n["in_place"] = False
            n["out_var"] = f"{n['id']}_out"
        new_nodes.append(n)
    return {"graph_id": graph["graph_id"], "nodes": new_nodes}
