import ref


def check(workdir):
    from compressor.tradeoff import evaluate_tradeoff

    out = {"tradeoff_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.evaluate_tradeoff(cfg)
        got = evaluate_tradeoff(cfg)
        if got and all(abs(got.get(k, 0) - want[k]) < 1e-5 for k in want):
            ok += 1
    out["tradeoff_matched"] = float(ok)
    return out
