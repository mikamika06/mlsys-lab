import numpy as np


def fold_constants(graph):
    initializers = dict(graph.get("initializers", {}))
    new_nodes = []
    for node in graph["nodes"]:
        if node["op"] == "Add" and all(inp in initializers for inp in node["inputs"]):
            vals = [initializers[inp] for inp in node["inputs"]]
            res = vals[0] + vals[1]
            for out in node["outputs"]:
                initializers[out] = res
        elif node["op"] == "Mul" and all(inp in initializers for inp in node["inputs"]):
            vals = [initializers[inp] for inp in node["inputs"]]
            res = vals[0] * vals[1]
            for out in node["outputs"]:
                initializers[out] = res
        else:
            new_nodes.append(node)
    return {**graph, "nodes": new_nodes, "initializers": initializers}
