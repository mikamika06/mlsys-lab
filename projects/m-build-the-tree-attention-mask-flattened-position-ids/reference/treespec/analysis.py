def analyze_sweep(sweep_records):
    sums = {}
    counts = {}
    for r in sweep_records:
        w = r["tree_width"]
        sums[w] = sums.get(w, 0.0) + r["accepted_length"]
        counts[w] = counts.get(w, 0) + 1
        
    return {w: sums[w] / counts[w] for w in sums}
