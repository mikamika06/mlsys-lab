import ref


def check(workdir):
    from kvmetric.calc import build_promql

    out = {"queries_match": 0.0}
    ok = 0
    for name, mtype, win in ref.QUERY_TESTS:
        want = ref.build_promql(name, mtype, win)
        try:
            got = build_promql(name, mtype, win)
        except Exception:
            got = ""
        if got.strip() == want.strip():
            ok += 1
    out["queries_match"] = float(ok)
    return out
