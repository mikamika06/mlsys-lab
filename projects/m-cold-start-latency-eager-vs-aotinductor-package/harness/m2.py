import ref


def check(workdir):
    from pack.latency import evaluate_cold_start

    out = {"latency_ratio_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want_ratio = ref.compute_latency_ratio(cfg)
        got_ratio = evaluate_cold_start(cfg)
        if abs(float(got_ratio) - float(want_ratio)) < 1e-4:
            ok += 1
    if ok == len(ref.CONFIGS):
        out["latency_ratio_matched"] = 1.0
    return out
