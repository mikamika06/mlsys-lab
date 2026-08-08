import ref


def check(workdir):
    from exportops.counter import count_aten_ops, identify_decompositions
    out = {"ops_matched": 0.0}
    ok = 0

    for g in ref.GRAPHS:
        want_counts = {}
        for n in g:
            if n.get("op") == "call_function" and "aten." in str(n.get("target", "")):
                t = n["target"]
                want_counts[t] = want_counts.get(t, 0) + 1

        got_counts = count_aten_ops(g)

        target_ops = ["aten.linear.default", "torch.ops.higher_order.cond"]
        want_decomp = {
            "counts": {op: want_counts.get(op, 0) for op in target_ops},
            "fully_decomposed": all(want_counts.get(op, 0) == 0 for op in target_ops)
        }
        got_decomp = identify_decompositions(g, target_ops)

        if got_counts == want_counts and got_decomp == want_decomp:
            ok += 1

    if ok == len(ref.GRAPHS):
        out["ops_matched"] = 1.0
    else:
        out["_note"] = f"matched {ok} of {len(ref.GRAPHS)}"

    return out
