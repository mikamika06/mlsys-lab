import os
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    sys.path.insert(0, os.path.dirname(__file__))

    import ref
    from cache.eviction import CacheSimulator

    trace = ref.generate_synthetic_trace(num_requests=800, vocab_size=80, zipf_alpha=1.3, seed=456)
    capacity = 25

    expected_lru = ref.ref_simulate_cache(capacity, trace, policy="lru")
    expected_lfu = ref.ref_simulate_cache(capacity, trace, policy="lfu")

    out = {"eviction_hit_rate_match": 0.0, "eviction_evicts_match": 0.0}
    try:
        sim_lru = CacheSimulator(capacity=capacity, policy="lru")
        for req in trace:
            sim_lru.access(req["key"], req["cost"])
        stats_lru = sim_lru.get_stats()

        sim_lfu = CacheSimulator(capacity=capacity, policy="lfu")
        for req in trace:
            sim_lfu.access(req["key"], req["cost"])
        stats_lfu = sim_lfu.get_stats()
    except Exception:
        return out

    lru_hr_ok = abs(stats_lru.get("hit_rate", -1) - expected_lru["hit_rate"]) < 1e-5
    lfu_hr_ok = abs(stats_lfu.get("hit_rate", -1) - expected_lfu["hit_rate"]) < 1e-5
    if lru_hr_ok and lfu_hr_ok:
        out["eviction_hit_rate_match"] = 1.0

    if stats_lru.get("evictions") == expected_lru["evictions"] and stats_lfu.get("evictions") == expected_lfu["evictions"]:
        out["eviction_evicts_match"] = 1.0

    return out
