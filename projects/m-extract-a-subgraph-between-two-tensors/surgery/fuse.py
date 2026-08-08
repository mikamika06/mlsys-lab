"""Operator fusion module."""
import copy

def fuse_gelu(graph):
    nodes = list(graph["nodes"])
    initializers = graph["initializer"]
    new_nodes = []
    match_found = False
    for node in nodes:
        if node["op_type"] == "Erf":
            gelu_node = {
                "name": node["name"] + "_gelu",
                "op_type": "Gelu",
                "inputs": [node["inputs"][0]],
                "outputs": node["outputs"]
            }
            new_nodes.append(gelu_node)
            match_found = True
        else:
            new_nodes.append(node)
    return {
        "nodes": new_nodes,
        "initializer": initializers,
        "input": graph["input"],
        "output": graph["output"]
    }
