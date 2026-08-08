def propagate_toy_dependencies(graph_nodes, pruned_indices, sparsity_ratio):
    active = set(range(graph_nodes))
    for idx in pruned_indices:
        if idx in active:
            active.remove(idx)
    propagated = sorted(list(active))
    return propagated
