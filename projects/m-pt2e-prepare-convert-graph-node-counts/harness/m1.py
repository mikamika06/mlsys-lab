import ref


def check(workdir):
    from pt2e_counts.analyzer import analyze_graph_counts
    out = {"counts_matched": 0.0, "total_configs": float(len(ref.GRAPHS))}
    ok = 0
    for i, item in enumerate(ref.GRAPHS):
        g = item["orig"]
        want = ref.analyze_graph_counts(g)
        got = analyze_graph_counts(g)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"graph {i}: got {got}, reference {want}"
    out["counts_matched"] = float(ok)
    return out
