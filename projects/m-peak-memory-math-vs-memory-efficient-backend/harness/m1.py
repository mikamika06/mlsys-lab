import ref


def check(workdir):
    from attnlab.memory import estimate_peak_memory

    out = {"memory_ratio_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        b, h, s, d = cfg["batch_size"], cfg["num_heads"], cfg["seq_len"], cfg["head_dim"]
        math_ref = ref.estimate_math_peak_memory(b, h, s, d)
        eff_ref = ref.estimate_efficient_peak_memory(b, h, s, d)
        math_got = estimate_peak_memory(b, h, s, d, backend="math")
        eff_got = estimate_peak_memory(b, h, s, d, backend="efficient")
        if abs(math_got - math_ref) < 1e-5 and abs(eff_got - eff_ref) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got math={math_got}, eff={eff_got}, want math={math_ref}, eff={eff_ref}"
    out["memory_ratio_matched"] = float(ok)
    return out
