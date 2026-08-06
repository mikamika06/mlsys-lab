import ref


def check(workdir):
    from tvmutils.analysis import derive_split_reorder_extents

    out = {"extents_matched": 0.0}
    ok = 0
    for tc in ref.TEST_CASES:
        try:
            got = derive_split_reorder_extents(tc["extent"], tc["factor"], tc["reorder"])
            outer_ref = (tc["extent"] + tc["factor"] - 1) // tc["factor"]
            inner_ref = tc["factor"]
            base = [outer_ref, inner_ref]
            want = [base[i] for i in tc["reorder"]]
            if got == want:
                ok += 1
        except Exception:
            pass
    out["extents_matched"] = float(ok)
    return out
