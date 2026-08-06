import ref


def check(workdir):
    from graphfix.graph_sweep import sweep_dead_and_orphans
    from graphfix.metrics import compute_simplification_payoff

    out = {
        "nodes_swept": 0.0,
        "initializers_swept": 0.0,
        "payoff_matched": 0.0
    }

    g = ref.GRAPHS[0]
    want_swept = ref.sweep_dead_and_orphans(g)
    want_payoff = ref.compute_simplification_payoff(g, want_swept)

    try:
        got_swept = sweep_dead_and_orphans(g)
    except Exception as e:
        out["_note"] = f"sweep raised exception: {type(e).__name__}: {str(e)[:100]}"
        return out

    got_nodes = [n["name"] for n in got_swept.get("nodes", [])]
    want_nodes = [n["name"] for n in want_swept.get("nodes", [])]

    if got_nodes == want_nodes:
        out["nodes_swept"] = 1.0
    else:
        out["_note"] = f"nodes mismatch: got {got_nodes}, want {want_nodes}"

    got_inits = set(got_swept.get("initializers", {}).keys())
    want_inits = set(want_swept.get("initializers", {}).keys())

    if got_inits == want_inits:
        out["initializers_swept"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"initializers mismatch: got {got_inits}, want {want_inits}"

    try:
        got_payoff = compute_simplification_payoff(g, got_swept)
        if got_payoff == want_payoff:
            out["payoff_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"payoff mismatch: got {got_payoff}, want {want_payoff}"
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"payoff raised exception: {type(e).__name__}: {str(e)[:100]}"

    return out
