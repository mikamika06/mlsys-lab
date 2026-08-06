import ref


def check(workdir):
    from spec.sweep import sweep_draft_tokens

    out = {"optimum_found": 0.0, "sweep_count": 0.0}
    cfg = ref.CONFIGS[0]
    res = sweep_draft_tokens(cfg)
    if not isinstance(res, dict) or "optimal_tokens" not in res:
        out["_note"] = "sweep_draft_tokens must return a dict with optimal_tokens"
        return out
    ref_res = ref.find_optimum(cfg)
    out["sweep_count"] = float(len(res.get("sweep_results", {})))
    if res["optimal_tokens"] == ref_res["optimal_tokens"]:
        out["optimum_found"] = 1.0
    else:
        out["_note"] = f"got optimal tokens {res['optimal_tokens']}, want {ref_res['optimal_tokens']}"
    return out
