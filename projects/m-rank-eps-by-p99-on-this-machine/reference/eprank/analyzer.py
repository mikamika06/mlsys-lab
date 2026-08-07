def rank_eps(latencies):
    p99_dict = {}
    for ep, lats in latencies.items():
        sorted_lats = sorted(lats)
        idx = int(0.99 * (len(sorted_lats) - 1))
        p99_dict[ep] = sorted_lats[idx]
    ranked = sorted(p99_dict.keys(), key=lambda k: (p99_dict[k], k))
    return ranked
