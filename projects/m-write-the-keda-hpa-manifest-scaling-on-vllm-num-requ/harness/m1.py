import ref


def check(workdir):
    from keda.parser import get_prometheus_query

    out = {"queries_matched": 0.0}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_query(cfg)
        got = get_prometheus_query(cfg)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["queries_matched"] = float(ok)
    return out
