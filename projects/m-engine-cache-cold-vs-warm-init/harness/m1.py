import ref


def check(workdir):
    from trt_engine.cache import verify_engine_cache, compute_warm_init_latency

    out = {"cache_matches": 0.0, "latency_ratio": 999.0, "_note": ""}
    ok = 0
    for i, item in enumerate(ref.CONFIGS):
        res = verify_engine_cache(item["cache_meta"], item["runtime_config"])
        if res == item["expected_valid"]:
            ok += 1
    out["cache_matches"] = float(ok)

    cold_latencies = [15.2, 15.1, 15.3]
    warm_latencies = [2.1, 2.0, 2.2]
    c_avg = compute_warm_init_latency(cold_latencies)
    w_avg = compute_warm_init_latency(warm_latencies)
    if w_avg > 0:
        out["latency_ratio"] = float(c_avg / w_avg)
    else:
        out["latency_ratio"] = 1.0

    return out
