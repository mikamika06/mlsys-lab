import torch
import torch.fx as fx

def count_fx_nodes(model, x):
    gm = fx.symbolic_trace(model)
    counts = {}
    for node in gm.graph.nodes:
        op = node.op
        counts[op] = counts.get(op, 0) + 1
    return counts
