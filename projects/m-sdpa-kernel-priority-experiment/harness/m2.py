import ref

def check(workdir):
    from sdpa_exp.memory import compute_memory_blowup
    out = {"blowup_match": 0.0}
    ok = True
    for cfg in ref.CONFIGS:
        want = ref.compute_memory_blowup(cfg)
        got = compute_memory_blowup(cfg)
        if abs(got - want) > 1e-5:
            ok = False
    out["blowup_match"] = 1.0 if ok else 0.0
    return out
