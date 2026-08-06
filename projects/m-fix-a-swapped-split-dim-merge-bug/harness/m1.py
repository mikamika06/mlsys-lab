import ref

def check(workdir):
    from ucp.mapping import reconstruct_mapping
    out = {"mappings_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.reconstruct_mapping(cfg)
        got = reconstruct_mapping(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    if ok == len(ref.CONFIGS):
        out["mappings_matched"] = 1.0
    return out
