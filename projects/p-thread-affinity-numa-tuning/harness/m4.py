import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from numa_tuning import allocator

    m = {"ratio_optimized": 0.0}
    try:
        res = allocator.optimize_instance_ratio(32, 2)
        if isinstance(res, dict) and res.get("threads_per_instance") == 16:
            m["ratio_optimized"] = 1.0
    except Exception:
        pass
    return m
