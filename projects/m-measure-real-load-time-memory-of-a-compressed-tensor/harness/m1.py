import ref


def check(workdir):
    from compress.measure import parse_metadata

    out = {"metadata_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.parse_metadata(cfg)
        got = parse_metadata(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["metadata_matched"] = float(ok)
    return out
