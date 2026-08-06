import ref


def check(workdir):
    from kvmetric.calc import hit_rate

    out = {"hit_rate_match": 0.0}
    ok = 0
    for prev_s, curr_s in ref.SCRAPES:
        want = ref.hit_rate(prev_s, curr_s)
        try:
            got = hit_rate(prev_s, curr_s)
        except Exception:
            got = -1.0
        if abs(got - want) < 1e-6:
            ok += 1
    out["hit_rate_match"] = float(ok)
    return out
