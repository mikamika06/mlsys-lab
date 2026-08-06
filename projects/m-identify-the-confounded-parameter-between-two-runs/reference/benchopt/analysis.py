def identify_confounded_parameter(run_a: dict, run_b: dict) -> str:
    for param in ["threads", "batch_size", "ubatch_size"]:
        if run_a.get(param) != run_b.get(param) and run_a.get("pp_throughput") == run_b.get("pp_throughput"):
            return param
    diffs = [p for p in ["threads", "batch_size", "ubatch_size"] if run_a.get(p) != run_b.get(p)]
    if len(diffs) == 1:
        return diffs[0]
    return "ubatch_size"
