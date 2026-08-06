import ref

def check(workdir):
    from moefit.quant import find_min_quant_bits
    out = {"bits_matched": 0.0, "models": float(len(ref.MODELS))}
    ok = 0
    for i, spec in enumerate(ref.MODELS):
        want = ref.min_bits(spec)
        got = find_min_quant_bits(spec, 36 * 1024 * 1024 * 1024)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: got {got}, reference {want}"
    out["bits_matched"] = float(ok)
    return out
