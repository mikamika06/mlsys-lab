import ref


def check(workdir):
    from roofline.model import compute_prefill_flops

    out = {"flops_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_prefill_flops(cfg)
        try:
            got = compute_prefill_flops(cfg)
        except Exception:
            got = -1
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["flops_matched"] = float(ok)
    return out
