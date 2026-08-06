import ref

def check(workdir):
    from diag.perf import compare_latencies
    _, perf_cases, _ = ref.generate_cases()
    correct = 0
    for seq, batch, expected_ratio in perf_cases:
        res = compare_latencies(seq, batch)
        if abs(res - expected_ratio) < 1e-5:
            correct += 1
    match = 1.0 if correct == len(perf_cases) else 0.0
    out = {"ratio_match": match}
    if match == 0.0:
        out["_note"] = f"Expected ratio {expected_ratio}, got {res}"
    return out
