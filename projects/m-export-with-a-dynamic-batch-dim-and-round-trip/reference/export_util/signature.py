import torch

def reconstruct_signature(exported_program):
    graph_module = exported_program.graph_module
    sig = {}
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            sig[node.name] = node.meta.get("val", None)
    return sig
