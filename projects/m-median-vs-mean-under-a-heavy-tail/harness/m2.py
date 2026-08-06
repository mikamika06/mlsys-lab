import ref

def check(workdir):
    from bench.warmup import quantify_warmup_inflation

    out = {"warmup_inflation_matched": 0.0}
    for cfg in ref.WARMUP_CONFIGS:
        samples = ref.generate_heavy_tail_samples(cfg["seed"], cfg["size"], 0.02)
        got = quantify_warmup_inflation(samples, cfg["warmup"])
        want = ref.compute_warmup_inflation(samples, cfg["warmup"])
        if abs(got - want["inflation"]) > 1e-4:
            return out
    out["warmup_inflation_matched"] = 1.0
    return out
