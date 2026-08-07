def compute_reuse_histogram(trace, max_dist=100):
    last_seen = {}
    dists = [0] * (max_dist + 1)
    for t, ev in enumerate(trace):
        bid = ev["block_id"]
        if bid in last_seen:
            d = t - last_seen[bid]
            if d <= max_dist:
                dists[d] += 1
            else:
                dists[max_dist] += 1
        last_seen[bid] = t
    return dists
