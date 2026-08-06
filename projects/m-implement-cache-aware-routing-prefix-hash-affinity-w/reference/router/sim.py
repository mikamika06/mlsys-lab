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
            from router.affinity import select_replica
            chosen = select_replica(replicas, ph, max_load_diff=100)
        elif policy == "cache_aware_guardrail":
            from router.affinity import select_replica
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
