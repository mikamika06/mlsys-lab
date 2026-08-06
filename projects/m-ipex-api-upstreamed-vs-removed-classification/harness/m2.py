import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    try:
        from ipexaudit.graph import analyze_layout_conversions, diff_op_graphs
    except Exception as e:
        return {"diff_matched": 0.0, "_note": f"Import failed: {e}"}

    out = {"diff_matched": 0.0}

    want_copies = ref.analyze_layout_conversions(ref.MANUAL_GRAPH)
    try:
        got_copies = analyze_layout_conversions(ref.MANUAL_GRAPH)
    except Exception as e:
        out["_note"] = f"analyze_layout_conversions raised {e}"
        return out

    if got_copies != want_copies:
        out["_note"] = f"analyze_layout_conversions got {got_copies}, expected {want_copies}"
        return out

    want_diff = ref.diff_op_graphs(ref.MANUAL_GRAPH, ref.IPEX_GRAPH)
    try:
        got_diff = diff_op_graphs(ref.MANUAL_GRAPH, ref.IPEX_GRAPH)
    except Exception as e:
        out["_note"] = f"diff_op_graphs raised {e}"
        return out

    if got_diff == want_diff:
        out["diff_matched"] = 1.0
    else:
        out["_note"] = f"diff_op_graphs got {got_diff}, expected {want_diff}"

    return out
