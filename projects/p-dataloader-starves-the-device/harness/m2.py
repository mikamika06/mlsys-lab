def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from dl.loader import compute_min_workers

    m = {"workers_optimal": 0.0}
    try:
        res = compute_min_workers(1.0, 0.2, 0.1)
        if res >= 4:
            m["workers_optimal"] = 1.0
    except Exception:
        pass
    return m
