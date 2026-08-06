import ref

def check(workdir):
    from feasibility.quant import evaluate_fp8_kv
    out = {"quant_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.QUANT_CONFIGS):
        want_gain, want_risk = ref.quant_gain_and_risk(cfg["fp16_bytes"], cfg["fp8_bytes"])
        got = evaluate_fp8_kv(cfg["fp16_bytes"], cfg["fp8_bytes"])
        if isinstance(got, tuple) and len(got) == 2:
            g_gain, g_risk = got
            if abs(g_gain - want_gain) < 1e-5 and isinstance(g_risk, str) and len(g_risk) > 0:
                ok += 1
        elif "_note" not in out:
            out["_note"] = f"quant config {i}: got {got}"
    out["quant_matched"] = float(ok)
    return out
