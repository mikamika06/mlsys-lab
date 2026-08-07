import ref


def check(workdir):
    from kvmodel.sizing import compute_kv_bytes

    out = {"bytes_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_kv_bytes(cfg, 128 * 1024, 1, 2)
        got = compute_kv_bytes(cfg, 128 * 1024, 1, 2)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["bytes_matched"] = float(ok)
    return out
