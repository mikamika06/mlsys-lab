import ref

def check(workdir):
    from relays import model, compare
    out = {"counts_matched": 0.0}
    try:
        mod = model.make_3op_model()
        if mod is None:
            out["_note"] = "make_3op_model returned None"
            return out
        got = compare.compare_ir_counts(mod)
        if got is None:
            out["_note"] = "compare_ir_counts returned None"
            return out
        want = ref.get_reference_counts()
        if got.get("relay_count") == want.get("relay_count") and got.get("relax_count") == want.get("relax_count"):
            out["counts_matched"] = 1.0
        else:
            out["_note"] = f"got counts {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {str(e)[:120]}"
    return out
