import ref


def check(workdir):
    from ir_compare.discrepancy import compute_folding_discrepancy

    out = {"discrepancy_matched": 0.0}
    ok = True
    for m in ref.MODELS:
        name = m["name"]
        want = ref.get_folding_discrepancy(name)
        got = compute_folding_discrepancy(name)
        if not got or abs(got.get("diff", -1) - want["diff"]) > 1e-6:
            ok = False
            out["_note"] = f"model {name} discrepancy: got {got}, reference {want}"
            break
    if ok:
        out["discrepancy_matched"] = 1.0
    return out
