import ref

def check(workdir):
    from bench.metrics import robust_central_tendency

    out = {"median_accuracy": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        samples = ref.generate_heavy_tail_samples(cfg["seed"], cfg["size"], cfg["tail_prob"])
        got = robust_central_tendency(samples)
        want, _ = ref.compute_median_vs_mean(samples)
        if abs(got - want) < 1e-5:
            ok += 1
    out["median_accuracy"] = float(ok)
    return out
