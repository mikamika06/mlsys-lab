import ref


def check(workdir):
    from ipexdiff.bench import evaluate_latency_and_speedup
    from ipexdiff.meta import get_ipex_deprecation_info

    out = {"benchmarks_correct": 0.0}

    res = evaluate_latency_and_speedup(ref.mock_model_runner, [1.0, 2.0, 3.0], num_runs=10)
    if not isinstance(res, dict):
        out["_note"] = "evaluate_latency_and_speedup must return a dictionary"
        return out

    for key in ("autocast_latency_ms", "ipex_latency_ms", "speedup"):
        if key not in res:
            out["_note"] = f"Missing key in bench results: {key}"
            return out

    if res["speedup"] <= 0:
        out["_note"] = f"Speedup ratio must be positive, got {res['speedup']}"
        return out

    meta = get_ipex_deprecation_info()
    if not isinstance(meta, dict):
        out["_note"] = "get_ipex_deprecation_info must return a dictionary"
        return out

    if meta.get("ipex_eol_date") != "2025-12-31":
        out["_note"] = f"Incorrect IPEX EOL date: {meta.get('ipex_eol_date')}"
        return out

    if meta.get("pytorch_upstream_version") != "2.6.0":
        out["_note"] = (
            f"Incorrect PyTorch upstream version: {meta.get('pytorch_upstream_version')}"
        )
        return out

    out["benchmarks_correct"] = 1.0
    return out
