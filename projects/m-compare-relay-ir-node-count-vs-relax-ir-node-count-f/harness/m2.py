import ref

def check(workdir):
    from relays import model, fold
    out = {"discrepancy_matched": 0.0}
    try:
        mod = model.make_3op_model()
        if mod is None:
            out["_note"] = "make_3op_model returned None"
            return out
        got = fold.check_constant_folding(mod)
        if got is None:
            out["_note"] = "check_constant_folding returned None"
            return out
        want = ref.get_reference_folding()
        if got.get("discrepancy") == want.get("discrepancy"):
            out["discrepancy_matched"] = 1.0
        else:
            out["_note"] = f"got folding {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {str(e)[:120]}"
    return out
