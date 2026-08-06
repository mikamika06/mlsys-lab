import ref

def check(workdir):
    from quantlib.crossover import find_crossover
    out = {"crossover_match": 0.0}
    want = ref.find_crossover(ref.SCHEMES, ref.WORKLOADS)
    try:
        got = find_crossover(ref.SCHEMES, ref.WORKLOADS)
    except Exception:
        got = None
    if got and want and got.get("workload_id") == want.get("workload_id"):
        if abs(got.get("throughput_ratio", 0) - want.get("throughput_ratio", 0)) < 1e-5:
            out["crossover_match"] = 1.0
    return out
