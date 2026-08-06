import ref


def check(workdir):
    from graphfix.fold_finder import find_frozen_dynamic_folds

    out = {"graphs_matched": 0.0, "total_graphs": float(len(ref.GRAPHS))}
    ok = 0
    for i, g in enumerate(ref.GRAPHS):
        want = ref.find_frozen_dynamic_folds(g)
        try:
            got = find_frozen_dynamic_folds(g)
        except Exception:
            got = None

        if got is not None and sorted(got) == sorted(want):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"graph {i}: got {got}, reference {want}"

    out["graphs_matched"] = float(ok)
    return out
