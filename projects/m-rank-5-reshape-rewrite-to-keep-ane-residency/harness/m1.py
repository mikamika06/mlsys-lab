import ref


def check(workdir):
    from anerewrite.rewrite import rewrite_shape
    out = {"rewrites_matched": 0.0}
    ok = 0
    for case in ref.REWRITE_CASES:
        want = ref.rewrite_shape(case)
        got = rewrite_shape(case)
        if got == want:
            ok += 1
    out["rewrites_matched"] = float(ok)
    return out
