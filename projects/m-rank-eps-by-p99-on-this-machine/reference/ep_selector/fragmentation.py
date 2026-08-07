import numpy as np


def compute_fragmentation_cost(subgraph_nodes, trt_supported_nodes):
    total = len(subgraph_nodes)
    if total == 0:
        return 0.0
    supported = set(trt_supported_nodes)
    is_supported = np.array([1 if n in supported else 0 for n in subgraph_nodes])
    switches = np.sum(np.abs(np.diff(is_supported)))
    fraction_unsupported = 1.0 - (np.sum(is_supported) / total)
    cost = float(switches * 2.5 + fraction_unsupported * 10.0)
    return cost
