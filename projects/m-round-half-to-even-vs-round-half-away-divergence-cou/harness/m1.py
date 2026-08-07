import ref

def check(workdir):
    from quantexport.divergence import count_divergences
    out = {"divergence_matched": 0.0}
    ok = 0
    for case in ref.CASES_DIV:
        want = ref.ref_count_divergences(case)
        got = count_divergences(case)
        if got == want:
            ok += 1
    out["divergence_matched"] = float(ok)
    return out
