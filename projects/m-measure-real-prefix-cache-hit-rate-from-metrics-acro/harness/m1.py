import ref


def check(workdir):
    from apcmetric.parser import parse_metrics

    out = {"metrics_matched": 0.0}
    ok = 0
    for text in ref.CONFIGS:
        want = ref.parse_metrics(text)
        got = parse_metrics(text)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"parsed mismatch: got {got}, want {want}"
    out["metrics_matched"] = float(ok)
    return out
