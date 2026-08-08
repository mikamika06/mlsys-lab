"""Constant folding module."""
import numpy as np
import copy

def fold_constants(graph):
    nodes = list(graph["nodes"])
    initializers = dict(graph["initializer"])
    changed = True
    while changed:
        changed = False
        new_nodes = []
        for node in nodes:
            op = node["op_type"]
            inp_names = node["inputs"]
            out_names = node["outputs"]
            if all(inp in initializers for inp in inp_names):
                vals = [initializers[inp] for inp in inp_names]
                if op == "Add":
                    res = vals[0] + vals[1]
                elif op == "Mul":
                    res = vals[0] * vals[1]
                elif op == "Sub":
                    res = vals[0] - vals[1]
                elif op == "Div":
                    res = vals[0] / vals[1]
                else:
                    res = None
                if res is not None:
                    for out_name in out_names:
                        initializers[out_name] = res
                    changed = True
                    continue
            new_nodes.append(node)
        nodes = new_nodes
    return {
        "nodes": nodes,
        "initializer": initializers,
        "input": graph["input"],
        "output": graph["output"]
    }
