def compute_metrics(baseline_outputs, runner_outputs):
    b_tokens = baseline_outputs.get("tokens", [])
    metrics = []
    for ro in runner_outputs:
        r_tokens = ro.get("tokens", [])
        rel_errors = []
        for bt, rt in zip(b_tokens, r_tokens):
            if bt == 0:
                rel_errors.append(0.0)
            else:
                rel_errors.append(abs(rt - bt) / float(bt))
        max_rel_err = max(rel_errors) if rel_errors else 0.0
        metrics.append({
            "name": ro.get("name"),
            "max_rel_err": max_rel_err,
            "throughput_ratio": sum(r_tokens) / float(sum(b_tokens)) if sum(b_tokens) > 0 else 0.0
        })
    return metrics
