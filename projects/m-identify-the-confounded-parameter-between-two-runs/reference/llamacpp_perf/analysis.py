def identify_confounded(run_a, run_b):
    diffs = []
    for k in set(run_a.keys()).union(run_b.keys()):
        if run_a.get(k) != run_b.get(k):
            diffs.append(k)
    if len(diffs) == 1:
        return diffs[0]
    for k in sorted(diffs):
        if k in ("threads", "batch_size", "ubatch_size", "-t", "-b", "-ub"):
            return k
    return diffs[0] if diffs else None

def tune_pp(baseline, candidates):
    base_pp = baseline.get("pp", 0.0)
    best = None
    best_pp = base_pp
    for c in candidates:
        if c.get("pp", 0.0) > best_pp:
            best_pp = c.get("pp", 0.0)
            best = c
    return best if best is not None else baseline

def check_tg_bytes_order(tg_runs, bytes_runs):
    sorted_tg = sorted(tg_runs, key=lambda x: x.get("id", 0))
    sorted_bytes = sorted(bytes_runs, key=lambda x: x.get("id", 0))
    tg_vals = [r.get("tg", 0.0) for r in sorted_tg]
    byte_vals = [r.get("bytes", 0.0) for r in sorted_bytes]
    tg_order = sorted(range(len(tg_vals)), key=lambda i: tg_vals[i])
    byte_order = sorted(range(len(byte_vals)), key=lambda i: byte_vals[i])
    return tg_order == byte_order
