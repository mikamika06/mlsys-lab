import numpy as np

def apply_transformer_fusion(model_graph):
    nodes = list(model_graph.get("nodes", []))
    new_nodes = []
    fused_count = 0
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node.get("op") == "AttentionSubGraph":
            fused_count += 1
            new_nodes.append({
                "name": f"FusedAttention_{fused_count}",
                "op": "FusedMultiHeadAttention",
                "inputs": node.get("inputs"),
                "outputs": node.get("outputs")
            })
            i += 1
        else:
            new_nodes.append(node)
            i += 1
    return {"nodes": new_nodes, "fused_count": fused_count}
