def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from kvtier import cost
    import ref

    m = {"cost_monotonic": 0.0, "transfer_time_ok": 0.0}

    c1 = cost.estimate_transfer_cost(100, 64, 10.0)
    c2 = cost.estimate_transfer_cost(200, 64, 10.0)
    oc1 = ref.oracle_transfer_cost(100, 64, 10.0)
    oc2 = ref.oracle_transfer_cost(200, 64, 10.0)

    if c2 > c1 and abs(c1 - oc1) < 1e-6 and abs(c2 - oc2) < 1e-6:
        m["cost_monotonic"] = 1.0
        m["transfer_time_ok"] = 1.0

    return m
