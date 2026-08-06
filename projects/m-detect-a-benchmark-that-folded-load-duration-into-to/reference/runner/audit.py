def parse_benchmark(data):
    return {
        "id": data.get("id"),
        "total_tokens": int(data.get("total_tokens", 0)),
        "generation_duration": float(data.get("generation_duration", 0.0)),
        "load_duration": float(data.get("load_duration", 0.0)),
        "reported_tok_s": float(data.get("reported_tok_s", 0.0))
    }

def is_load_folded(data, tol=1e-2):
    p = parse_benchmark(data)
    if p["generation_duration"] <= 0 or p["total_tokens"] <= 0:
        return False
    pure = p["total_tokens"] / p["generation_duration"]
    total_time = p["generation_duration"] + p["load_duration"]
    folded = p["total_tokens"] / total_time if total_time > 0 else 0.0
    diff_pure = abs(p["reported_tok_s"] - pure)
    diff_folded = abs(p["reported_tok_s"] - folded)
    if p["load_duration"] > 1e-4 and diff_folded < diff_pure and diff_folded < tol * pure:
        return True
    return False

def detect_folded_benchmarks(benchmarks):
    bad = []
    for b in benchmarks:
        if is_load_folded(b):
            bad.append(b.get("id"))
    return bad
