import ref

def check(workdir):
    from loratool.sizing import adapter_size_bytes

    out = {"size_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_adapter_bytes(cfg)
        got = adapter_size_bytes(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["size_matched"] = 1.0 if ok == len(ref.CONFIGS) else 0.0
    return out
