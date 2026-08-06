import ref


def check(workdir):
    from peftutils.size import compute_adapter_bytes

    out = {"sizes_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        for targets in ref.TARGET_MODULES:
            want = ref.compute_adapter_bytes(cfg, targets)
            got = compute_adapter_bytes(cfg, targets)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"config {i}: got {got}, reference {want}"
    out["sizes_matched"] = float(ok)
    return out
