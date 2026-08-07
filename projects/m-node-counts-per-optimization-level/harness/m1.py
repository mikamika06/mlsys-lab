import ref


def check(workdir):
    from ortopt.analyzer import count_nodes

    out = {"counts_matched": 0.0}
    ok = 0
    levels = [0, 1, 99]
    total = len(ref.MODELS) * len(levels)
    matched = 0
    for m in ref.MODELS:
        for lvl in levels:
            want = ref.count_nodes(m, lvl)
            got = count_nodes(m, lvl)
            if want == got:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"model node count mismatch at level {lvl}: got {got}, want {want}"
    out["counts_matched"] = float(matched)
    return out
