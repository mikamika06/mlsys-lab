import ref


def check(workdir):
    from otel_cache.attribution import extract_attribution

    out = {"requests_matched": 0.0, "total": float(len(ref.CONFIGS))}
    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.extract_attribution(cfg["spans"])
        got = extract_attribution(cfg["spans"])
        if got == want:
            ok += 1
    out["requests_matched"] = float(ok)
    return out
