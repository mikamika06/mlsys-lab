def check(workdir):
    import ref
    from sparsity.engine import Device, get_speedup
    m = {"correct_ratio": 0.0}
    d = Device("mock", True, 1000.0, 10000.0)

    try:
        ratio = get_speedup(100, 100, 100, d)
        expected = ref.get_speedup(100, 100, 100, d)
        if abs(ratio - expected) < 1e-5:
            m["correct_ratio"] = 1.0
    except Exception:
        pass

    return m
