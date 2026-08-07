import ref

def check(workdir):
    from servebatch.effects import measure_effects
    out = {"effects_matched": 0.0}
    ok = True
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.measure_effects(cfg["arrivals"], cfg["max_batch"], cfg["timeout"], cfg["concurrencies"])
        got = measure_effects(cfg["arrivals"], cfg["max_batch"], cfg["timeout"], cfg["concurrencies"])
        if set(want.keys()) != set(got.keys()):
            ok = False
            out["_note"] = f"config {i}: concurrency keys mismatch"
            break
        for c in cfg["concurrencies"]:
            if abs(want[c] - got[c]) > 1e-5:
                ok = False
                out["_note"] = f"config {i}, concurrency {c}: got {got[c]}, want {want[c]}"
                break
    if ok:
        out["effects_matched"] = 1.0
    return out
