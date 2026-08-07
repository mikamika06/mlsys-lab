import ref

def check(workdir):
    from loratool.divergence import measure_divergence

    out = {"divergence_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_divergence(cfg)
        got = measure_divergence(cfg)
        if abs(got - want) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["divergence_matched"] = float(ok)
    return out
