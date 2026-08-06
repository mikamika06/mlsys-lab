import ref

def check(workdir):
    from affinity.compare import parse_and_compare
    logs = ref.generate_logs()
    want = ref.compute_latency_ratio(logs)
    got = parse_and_compare(logs)
    matched = 0
    for k, v in want.items():
        if k in got and abs(got[k] - v) < 1e-5:
            matched += 1
    return {"latency_ratio_matched": float(matched)}
