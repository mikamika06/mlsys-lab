from mlsys.sim import cache as cachesim

def grade(sol, fx) -> dict:
    configs = [
        # (n_nodes, node_size, line_bytes, sets, ways, mlp, miss_latency)
        (256,  128, 64, 16, 4, 4, 200),
        (512,   64, 64, 32, 8, 8, 100),
        (128,  256, 64,  8, 2, 2, 300),
        (512,   32, 64, 16, 4, 4, 150),
        (1024, 128, 64, 32, 8, 4, 100),
    ]

    max_err = 0.0
    for n_nodes, node_size, line_bytes, sets, ways, mlp, miss_latency in configs:
        addrs = [i * node_size for i in range(n_nodes)]
        ref_misses = cachesim.simulate(
            addrs, line_bytes=line_bytes, sets=sets, ways=ways
        )["misses"]
        ref = (ref_misses / mlp) * miss_latency
        try:
            ans = sol.modeled_runtime(
                n_nodes, node_size, line_bytes, sets, ways, mlp, miss_latency
            )
        except Exception:
            return {"rel_err": 1.0}
        if ref == 0.0:
            err = 0.0 if ans == 0.0 else 1.0
        else:
            err = abs(ans - ref) / ref
        max_err = max(max_err, err)

    return {"rel_err": max_err}
