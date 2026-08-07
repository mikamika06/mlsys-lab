def check(workdir):
    import ref
    from sparsity.engine import Device, simulate_time
    m = {"dense_time": 0.0, "sparse_time": 0.0}
    d = Device("mock", True, 1000.0, 10000.0)

    try:
        t_d = simulate_time(100, 100, 100, "dense", d)
        if abs(t_d - ref.simulate_time(100, 100, 100, "dense", d)) < 1e-5:
            m["dense_time"] = 1.0

        t_s = simulate_time(100, 100, 100, "sparse_2_4", d)
        if abs(t_s - ref.simulate_time(100, 100, 100, "sparse_2_4", d)) < 1e-5:
            m["sparse_time"] = 1.0
    except Exception:
        pass

    return m
