import ref


def check(workdir):
    from ir_compare.counts import get_node_counts

    out = {"counts_matched": 0.0}
    ok = True
    for m in ref.MODELS:
        name = m["name"]
        want = ref.get_reference_counts(name)
        got = get_node_counts(name)
        if got != want:
            ok = False
            out["_note"] = f"model {name}: got {got}, reference {want}"
            break
    if ok:
        out["counts_matched"] = 1.0
    return out
