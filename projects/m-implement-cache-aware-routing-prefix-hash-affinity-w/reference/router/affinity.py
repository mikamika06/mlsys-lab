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
