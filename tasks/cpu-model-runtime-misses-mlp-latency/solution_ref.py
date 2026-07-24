from mlsys.sim import cache as cachesim

def modeled_runtime(n_nodes, node_size, line_bytes, sets, ways, mlp, miss_latency):
    """Return the modeled execution time in cycles (float)."""
    addrs = [i * node_size for i in range(n_nodes)]
    misses = cachesim.simulate(
        addrs, line_bytes=line_bytes, sets=sets, ways=ways
    )["misses"]
    return (misses / mlp) * miss_latency
