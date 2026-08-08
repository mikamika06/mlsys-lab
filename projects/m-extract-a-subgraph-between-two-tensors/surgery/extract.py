"""Subgraph extraction module."""
import copy

def extract_subgraph(graph, input_tensor, output_tensor):
    nodes = graph["nodes"]
    initializers = graph["initializer"]
    producer = {}
    for node in nodes:
        for out in node["outputs"]:
            producer[out] = node
    included_nodes = []
    queue = [output_tensor]
    while queue:
        t = queue.pop(0)
        if t == input_tensor or t in initializers:
            continue
        if t in producer:
            node = producer[t]
            if node not in included_nodes:
                included_nodes.append(node)
                for inp in node["inputs"]:
                    if inp not in queue:
                        queue.append(inp)
    included_nodes.reverse()
    used_inits = {}
    for node in included_nodes:
        for inp in node["inputs"]:
            if inp in initializers:
                used_inits[inp] = initializers[inp]
    return {
        "nodes": included_nodes,
        "initializer": used_inits,
        "input": [{"name": input_tensor}],
        "output": [output_tensor]
    }
