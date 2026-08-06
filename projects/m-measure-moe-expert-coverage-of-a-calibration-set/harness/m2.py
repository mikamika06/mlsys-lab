import ref


def check(workdir):
    from moecov.compare import compare_imatrices

    out = {"comparison_matched": 0.0}
    ok = 0
    for i, (a, b) in enumerate(ref.COMPARISONS):
        want = ref.compare_imatrices(a, b)
        got = compare_imatrices(a, b)
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"comparison {i}: got {got}, reference {want}"
    out["comparison_matched"] = float(ok)
    return out
