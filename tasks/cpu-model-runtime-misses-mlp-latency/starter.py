def modeled_runtime(n_nodes, node_size, line_bytes, sets, ways, mlp, miss_latency):
    """Return the modeled execution time in cycles (float).

    Generate the pointer-chase trace for n_nodes sequential nodes of
    node_size bytes each (addresses 0, node_size, 2*node_size, ...),
    determine how many cache misses it produces in the given set-associative
    cache, and compute (misses / mlp) * miss_latency.
    """
    raise NotImplementedError('your code here')
