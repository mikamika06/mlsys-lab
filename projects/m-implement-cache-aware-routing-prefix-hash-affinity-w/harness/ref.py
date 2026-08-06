import numpy as np

def select_replica(replicas, prefix_hash, max_load_diff=3):
    best_r = -1
    best_score = -1.0
    min_load = min(r["load"] for r in replicas)
    for i, r in enumerate(replicas):
        if r["load"] > min_load + max_load_diff:
            continue
        cached = 1.0 if prefix_hash in r["cache"] else 0.0
        score = cached - 0.05 * r["load"]
        if score > best_score:
            best_score = score
            best_r = i
    if best_r == -1:
        best_r = min(range(len(replicas)), key=lambda x: replicas[x]["load"])
    return best_r

def simulate_trace(replicas_count, trace, policy):
    replicas = [{"load": 0, "cache": set()} for _ in range(replicas_count)]
    hits = 0
    total = len(trace)
    rr_idx = 0
    for req in trace:
        ph = req["prefix_hash"]
        if policy == "round_robin":
            chosen = rr_idx % replicas_count
            rr_idx += 1
        elif policy == "least_outstanding":
            chosen = min(range(replicas_count), key=lambda x: replicas[x]["load"])
        elif policy == "prefix_affinity":
            chosen = select_replica(replicas, ph, max_load_diff=100)
        elif policy == "cache_aware_guardrail":
            chosen = select_replica(replicas, ph, max_load_diff=2)
        else:
            chosen = 0
        if ph in replicas[chosen]["cache"]:
            hits += 1
        else:
            replicas[chosen]["cache"].add(ph)
        replicas[chosen]["load"] += 1
        replicas[chosen]["load"] = max(0, replicas[chosen]["load"] - 1)
    return hits / max(1, total)

def calc_replicas(lam, service_rate, max_queue):
    effective_rate = lam / service_rate
    req_rep = np.ceil(effective_rate + np.sqrt(max_queue))
    return int(max(1, req_rep))

def get_reference_trace():
    rng = np.random.default_rng(42)
    hashes = [rng.integers(0, 10) for _ in range(50)]
    return [{"prefix_hash": int(h)} for h in hashes]
