def optimize_min_cut(graph_nodes, memory_limit):
    cuts = []
    current_mem = 0
    for i, cost in enumerate(graph_nodes):
        current_mem += cost
        if current_mem > memory_limit:
            cuts.append(i)
            current_mem = cost
    return cuts
