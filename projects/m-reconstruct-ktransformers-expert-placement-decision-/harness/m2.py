import sys
import os

def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from ktrans.cache import simulate_lru_cache
    from ktrans.offload import evaluate_offload_latency

    trace = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5, 1, 6, 7, 2, 3, 1]
    cap = 3
    want_hit_rate = ref.simulate_lru_cache(cap, trace)
    got_hit_rate = simulate_lru_cache(cap, trace)

    cache_ok = 1 if abs(want_hit_rate - got_hit_rate) < 1e-6 else 0

    want_lat = ref.evaluate_offload_latency(32, 100, 0.002, 0.010, 0.005, 12)
    got_lat = evaluate_offload_latency(32, 100, 0.002, 0.010, 0.005, 12)

    lat_ok = 0
    if isinstance(got_lat, dict):
        diff1 = abs(want_lat["offload_all_latency"] - got_lat.get("offload_all_latency", 0.0))
        diff2 = abs(want_lat["offload_split_latency"] - got_lat.get("offload_split_latency", 0.0))
        diff3 = abs(want_lat["speedup"] - got_lat.get("speedup", 0.0))
        if diff1 < 1e-5 and diff2 < 1e-5 and diff3 < 1e-5:
            lat_ok = 1

    return {
        "cache_sim_matched": float(cache_ok),
        "latency_eval_matched": float(lat_ok)
    }
