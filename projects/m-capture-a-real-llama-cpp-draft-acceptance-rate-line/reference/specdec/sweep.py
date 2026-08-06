def find_optimal_draft_n_max(sweep_runs):
    best_run = None
    best_tps = -1.0
    results = []
    for run in sweep_runs:
        draft_n = run["spec_draft_n_max"]
        eval_time_ms = run.get("eval_time_ms", 0.0)
        total_tokens = run.get("generated_tokens", 0)
        tps = (total_tokens / (eval_time_ms / 1000.0)) if eval_time_ms > 0 else 0.0
        entry = {
            "spec_draft_n_max": draft_n,
            "tokens_per_second": tps,
            "accepted_tokens": run.get("accepted_tokens", 0),
            "sampled_tokens": run.get("sampled_tokens", 0)
        }
        results.append(entry)
        if tps > best_tps:
            best_tps = tps
            best_run = draft_n

    return {
        "optimal_draft_n_max": best_run,
        "max_tokens_per_second": best_tps,
        "sweep_details": results
    }
