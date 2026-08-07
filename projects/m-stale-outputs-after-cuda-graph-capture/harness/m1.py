import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    try:
        from ortgraph.legality import classify_pipeline, is_graph_legal
    except Exception as e:
        return {"legality_matched": 0.0, "_note": f"Import failed: {e}"}

    out = {"legality_matched": 0.0}
    ok = True

    for node in ref.PIPELINE_NODES:
        want_legal, want_reason = ref.is_graph_legal(node)
        try:
            got_legal, got_reason = is_graph_legal(node)
        except Exception as e:
            out["_note"] = f"is_graph_legal raised: {e}"
            return out

        if (got_legal, got_reason) != (want_legal, want_reason):
            ok = False
            out["_note"] = (
                f"Node {node.get('op_type')}: got ({got_legal}, {got_reason}), "
                f"expected ({want_legal}, {want_reason})"
            )
            break

    if ok:
        want_summary = ref.classify_pipeline(ref.PIPELINE_NODES)
        try:
            got_summary = classify_pipeline(ref.PIPELINE_NODES)
        except Exception as e:
            out["_note"] = f"classify_pipeline raised: {e}"
            return out

        if got_summary == want_summary:
            out["legality_matched"] = 1.0
        else:
            out["_note"] = f"Summary mismatch: got {got_summary}, expected {want_summary}"

    return out
