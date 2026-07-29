import workload as W


def check(workdir):
    spec = W.trace(6, 128, 8, 2, prefix=W.SHARED)
    cold, _ = W.learner(workdir, spec, dict(W.CFG_BASE, prefix_cache=False))
    warm, _ = W.learner(workdir, spec, W.CFG_CACHE)
    ref, _ = W.oracle(spec, W.CFG_CACHE)
    other = W.trace(4, 128, 8, 2, prefix=tuple(range(700, 764)))
    iso, _ = W.learner(workdir, spec + other, W.CFG_CACHE)
    riso, _ = W.oracle(spec + other, W.CFG_CACHE)
    return {
        "cache_saves_prefill": 1.0 if warm.get("prefill_tokens", 10 ** 9) < cold.get("prefill_tokens", 0) else 0.0,
        "hit_rate": float(warm.get("cache_hit_rate", 0.0)),
        "hit_rate_match": 1.0 if abs(warm.get("cache_hit_rate", -1) - ref["cache_hit_rate"]) < 1e-9 else 0.0,
        "mixed_tenants_match": 1.0 if iso.get("prefill_tokens") == riso["prefill_tokens"] else 0.0,
    }
