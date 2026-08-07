import ref


def check(workdir):
    from tensorsplit.sizes import compute_layer_sizes

    out = {"sizes_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_layer_sizes(cfg)
        got = compute_layer_sizes(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["sizes_matched"] = float(ok)
    return out
