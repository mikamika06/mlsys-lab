"""Reference oracle implementation for harness verification."""

import random


def generate_synthetic_trace(num_requests=1000, vocab_size=100, zipf_alpha=1.2, seed=42):
    rng = random.Random(seed)
    weights = [1.0 / (i ** zipf_alpha) for i in range(1, vocab_size + 1)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    cum_probs = []
    c = 0.0
    for p in probs:
        c += p
        cum_probs.append(c)

    trace = []
    for i in range(num_requests):
        r = rng.random()
        idx = 0
        for j, cp in enumerate(cum_probs):
            if r <= cp:
                idx = j
                break
        trace.append({
            "rid": f"req_{i}",
            "key": f"prompt_key_{idx}",
            "cost": 0.05 + (idx % 5) * 0.02,
        })
    return trace


def ref_analyze_trace(trace):
    if not trace:
        return {"total_requests": 0, "unique_keys": 0, "hits": 0, "hit_rate": 0.0}
    seen = set()
    hits = 0
    for req in trace:
        k = req["key"]
        if k in seen:
            hits += 1
        else:
            seen.add(k)
    total = len(trace)
    return {
        "total_requests": total,
        "unique_keys": len(seen),
        "hits": hits,
        "hit_rate": hits / total if total > 0 else 0.0,
    }


def ref_compute_net_savings(hit_rate, total_requests, compute_cost_per_req, memory_cost_per_entry):
    hits = hit_rate * total_requests
    compute_saved = hits * compute_cost_per_req
    net_savings = compute_saved - memory_cost_per_entry
    return {
        "compute_saved": compute_saved,
        "net_savings": net_savings,
        "is_profitable": net_savings > 0,
    }


def ref_calculate_memory_footprint(num_entries, avg_key_len_tokens, avg_val_len_tokens, bytes_per_token=2):
    bytes_per_entry = (avg_key_len_tokens + avg_val_len_tokens) * bytes_per_token
    total_bytes = num_entries * bytes_per_entry
    return {
        "bytes_per_entry": bytes_per_entry,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
    }


def ref_simulate_cache(capacity, trace, policy="lru"):
    cache = {}
    freqs = {}
    costs = {}
    access_time = {}
    timer = 0
    hits = 0
    misses = 0
    evictions = 0

    for req in trace:
        timer += 1
        key = req["key"]
        cost = req.get("cost", 1.0)
        if key in cache:
            hits += 1
            freqs[key] += 1
            access_time[key] = timer
        else:
            misses += 1
            if len(cache) >= capacity and capacity > 0:
                if policy == "lru":
                    victim = min(cache.keys(), key=lambda k: access_time[k])
                elif policy == "lfu":
                    victim = min(cache.keys(), key=lambda k: (freqs[k], access_time[k]))
                elif policy == "cost":
                    victim = min(cache.keys(), key=lambda k: (costs[k], -access_time[k]))
                else:
                    victim = next(iter(cache.keys()))
                del cache[victim]
                del freqs[victim]
                del costs[victim]
                del access_time[victim]
                evictions += 1

            if capacity > 0:
                cache[key] = True
                freqs[key] = 1
                costs[key] = cost
                access_time[key] = timer

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_rate": hit_rate,
        "size": len(cache),
    }


def ref_evaluate_cache_viability(capacity, trace, compute_cost_per_req, memory_cost_per_entry):
    sim_res = ref_simulate_cache(capacity, trace, policy="lru")
    total_reqs = len(trace)
    total_memory_cost = capacity * memory_cost_per_entry
    hits = sim_res["hits"]
    hit_rate = sim_res["hit_rate"]

    compute_saved = hits * compute_cost_per_req
    net_savings = compute_saved - total_memory_cost
    roi = (net_savings / total_memory_cost) if total_memory_cost > 0 else 0.0

    required_hits = total_memory_cost / compute_cost_per_req if compute_cost_per_req > 0 else 0
    breakeven_rate = min(1.0, max(0.0, required_hits / total_reqs)) if total_reqs > 0 else 1.0

    return {
        "capacity": capacity,
        "hit_rate": hit_rate,
        "compute_saved": compute_saved,
        "memory_cost": total_memory_cost,
        "net_savings": net_savings,
        "roi": roi,
        "breakeven_hit_rate": breakeven_rate,
        "should_enable": net_savings > 0 and hit_rate >= breakeven_rate,
    }
