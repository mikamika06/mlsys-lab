import ref


def check(workdir):
    out = {"fused_counts_matched": 0.0, "costs_matched": 0.0}
    try:
        from ortopt.costs import evaluate_offline_vs_online
        from ortopt.fused import count_fused_nodes
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {e}"
        return out

    graphs = ref.generate_graph_data()
    fused_ok = True
    for graph in graphs:
        want = ref.ref_count_fused_nodes(graph)
        try:
            got = count_fused_nodes(graph)
        except Exception as e:
            out["_note"] = f"count_fused_nodes error: {type(e).__name__}: {e}"
            return out
        if got != want:
            fused_ok = False
            out["_note"] = f"count_fused_nodes mismatch: expected {want}, got {got}"
            break

    if fused_ok:
        out["fused_counts_matched"] = 1.0

    cost_cases = ref.generate_cost_data()
    costs_ok = True
    for reqs, setup, per_req in cost_cases:
        want = ref.ref_evaluate_offline_vs_online(reqs, setup, per_req)
        try:
            got = evaluate_offline_vs_online(reqs, setup, per_req)
        except Exception as e:
            out["_note"] = f"evaluate_offline_vs_online error: {type(e).__name__}: {e}"
            return out
        if got != want:
            costs_ok = False
            out["_note"] = f"evaluate_offline_vs_online mismatch: expected {want}, got {got}"
            break

    if costs_ok:
        out["costs_matched"] = 1.0

    return out
