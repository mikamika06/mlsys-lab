def fuse_gelu(graph):
    new_nodes = []
    nodes = graph["nodes"]
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node["op"] == "Mul" and i + 3 < len(nodes):
            n1 = nodes[i]
            n2 = nodes[i+1]
            n3 = nodes[i+2]
            n4 = nodes[i+3]
            if (n2["op"] == "Erf" and n3["op"] == "Add" and n4["op"] == "Mul" and
                n2["inputs"][0] in n1["outputs"] and
                n3["inputs"][0] in n1["outputs"] and n3["inputs"][1] in n2["outputs"] and
                n4["inputs"][0] in n1["outputs"] and n4["inputs"][1] in n3["outputs"]):
                fused = {
                    "name": f"Gelu_{node['name']}",
                    "op": "Gelu",
                    "inputs": [n1["inputs"][0] if n1["inputs"][0] != n1["inputs"][1] else n1["inputs"][1]],
                    "outputs": n4["outputs"]
                }
                new_nodes.append(fused)
                i += 4
                continue
        new_nodes.append(node)
        i += 1
    return {**graph, "nodes": new_nodes}
