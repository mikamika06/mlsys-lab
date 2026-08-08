def fuse_gelu(graph):
    nodes = list(graph["nodes"])
    initializers = dict(graph["initializers"])

    i = 0
    new_nodes = []
    while i < len(nodes):
        if i + 3 < len(nodes):
            n1, n2, n3, n4 = nodes[i], nodes[i+1], nodes[i+2], nodes[i+3]
            if (n1["op"] == "Mul" and n2["op"] == "Erf" and n3["op"] == "Add" and n4["op"] == "Mul" and
                n1["outputs"][0] == n2["inputs"][0] and
                n2["outputs"][0] == n3["inputs"][0] and
                n3["outputs"][0] == n4["inputs"][1] and
                n1["inputs"][0] == n4["inputs"][0]):
                fused = {
                    "name": f"gelu_{n1['name']}",
                    "op": "Gelu",
                    "inputs": [n1["inputs"][0]],
                    "outputs": n4["outputs"],
                    "attributes": {}
                }
                new_nodes.append(fused)
                i += 4
                continue
        new_nodes.append(nodes[i])
        i += 1

    return {"nodes": new_nodes, "initializers": initializers, "inputs": graph["inputs"], "outputs": graph["outputs"]}
