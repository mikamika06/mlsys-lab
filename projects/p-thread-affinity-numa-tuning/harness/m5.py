import ref


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from numa_tuning import allocator

    m = {"high_scaling_ok": 0.0}
    try:
        data = ref.get_mock_scaling_data()
        ok = allocator.verify_high_scaling(data, 0.5)
        if ok:
            m["high_scaling_ok"] = 1.0
    except Exception:
        pass
    return m
