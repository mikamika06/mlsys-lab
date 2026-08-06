def fuse_bert_graph(graph):
    nodes = graph["nodes"]
    new_nodes = []
    fused = False
    for n in nodes:
        if n["op"] == "Attention":
            new_nodes.append({"name": "FusedAttention", "op": "FusedAttention", "inputs": n["inputs"]})
            fused = True
        else:
            new_nodes.append(n)
    return {"nodes": new_nodes, "fused": fused}
