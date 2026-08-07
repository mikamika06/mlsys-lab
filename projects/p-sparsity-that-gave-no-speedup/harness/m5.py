def check(workdir):
    from sparsity.engine import Device, check_if_speedup_possible
    m = {"mem_bound_no_speedup": 0.0, "compute_bound_speedup": 0.0}
    d_compute = Device("mock1", True, 1e9, 10.0)
    d_mem = Device("mock2", True, 10.0, 1e9)

    try:
        if check_if_speedup_possible(100, 100, 100, d_compute):
            m["compute_bound_speedup"] = 1.0

        if not check_if_speedup_possible(10000, 10, 10, d_mem):
            m["mem_bound_no_speedup"] = 1.0
    except Exception:
        pass

    return m
