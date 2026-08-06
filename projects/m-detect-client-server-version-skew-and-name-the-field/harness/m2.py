import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)

    from runnerdiag.binding import reconcile_host_binding
    from runnerdiag.latency import measure_first_request_cost

    out = {"binding_reconciled": 0.0}

    binding_cases = ref.generate_binding_test_cases()
    binding_ok = True
    for env, socks, expected in binding_cases:
        res = reconcile_host_binding(env, socks)
        if not isinstance(res, dict) or res.get("is_reconciled") != expected:
            binding_ok = False
            out["_note"] = f"Binding mismatch for env={env}, got {res}"
            break

    if not binding_ok:
        return out

    req_fn = ref.mock_request_runner()
    lat_res = measure_first_request_cost(req_fn, iterations=4)

    if (isinstance(lat_res, dict) and
        abs(lat_res.get("first_request_ms", 0.0) - 150.0) < 1e-3 and
        abs(lat_res.get("warm_avg_ms", 0.0) - 10.0) < 1e-3 and
        abs(lat_res.get("cold_start_overhead_ms", 0.0) - 140.0) < 1e-3):
        out["binding_reconciled"] = 1.0
    else:
        out["_note"] = f"Latency measurement discrepancy: got {lat_res}"

    return out
