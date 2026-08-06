import ref


def check(workdir):
    from isareport.contrast import contrast_isa
    from isareport.analyzer import has_mismatch

    out = {"contrast_match": 0.0, "mismatch_detected": 0.0}
    ok_contrast = 0
    ok_mismatch = 0
    for rep in ref.REPORTS:
        want_contrast = ref.contrast_reports(rep["native"], rep["manual"])
        got_contrast = contrast_isa(rep["native"], rep["manual"])
        if got_contrast == want_contrast:
            ok_contrast += 1

        want_mismatch = len(want_contrast) > 0
        got_mismatch = has_mismatch(rep["native"], rep["manual"])
        if bool(got_mismatch) == bool(want_mismatch):
            ok_mismatch += 1

    if ok_contrast == len(ref.REPORTS):
        out["contrast_match"] = 1.0
    if ok_mismatch == len(ref.REPORTS):
        out["mismatch_detected"] = 1.0
    return out
