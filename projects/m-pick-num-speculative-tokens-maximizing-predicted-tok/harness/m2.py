import ref

def check(workdir):
    from spec.draft import measure_acceptance
    out = {"acceptance_matched": 0.0}
    want = ref.evaluate_ngram_acceptance(ref.CODE_EDIT_TRACE)
    got = measure_acceptance(ref.CODE_EDIT_TRACE)
    if abs(got - want) < 1e-5:
        out["acceptance_matched"] = 1.0
    else:
        out["_note"] = f"got acceptance {got}, want {want}"
    return out
