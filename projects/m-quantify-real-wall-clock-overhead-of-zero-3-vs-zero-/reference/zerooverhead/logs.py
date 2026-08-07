def filter_warmup_steps(records, warmup_steps=5):
    return [r for r in records if r.get("step", 0) > warmup_steps]


def aggregate_stage_times(records):
    if not records:
        return {
            "fwd_compute_ms": 0.0,
            "bwd_compute_ms": 0.0,
            "param_gather_ms": 0.0,
            "grad_reduce_ms": 0.0,
            "opt_step_ms": 0.0,
            "total_step_ms": 0.0,
        }
    keys = [
        "fwd_compute_ms",
        "bwd_compute_ms",
        "param_gather_ms",
        "grad_reduce_ms",
        "opt_step_ms",
        "total_step_ms",
    ]
    res = {k: 0.0 for k in keys}
    n = float(len(records))
    for r in records:
        for k in keys:
            res[k] += float(r.get(k, 0.0))
    return {k: v / n for k, v in res.items()}
