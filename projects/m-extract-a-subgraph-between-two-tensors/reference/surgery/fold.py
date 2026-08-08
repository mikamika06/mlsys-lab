def fold_constants(graph):
    nodes = list(graph["nodes"])
    initializers = dict(graph["initializers"])

    folded_inits = dict(initializers)
    new_nodes = []
    for node in nodes:
        if node["op"] == "Add" and len(node["inputs"]) == 2:
            i0, i1 = node["inputs"]
            if i0 in folded_inits and i1 in folded_inits:
                res = folded_inits[i0] + folded_inits[i1]
                out_name = node["outputs"][0]
                folded_inits[out_name] = res
                continue
        new_nodes.append(node)

    active = set(graph["outputs"])
    for n in new_nodes:
        for inp in n["inputs"]:
            active.add(inp)
    for inp in graph["inputs"]:
        active.add(inp)

    cleaned_inits = {k: v for k, v in folded_inits.items() if k in active}
    return {"nodes": new_nodes, "initializers": cleaned_inits, "inputs": graph["inputs"], "outputs": graph["outputs"]}
