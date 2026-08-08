import ref

def check(workdir):
    from onednn.analysis import analyze_log
    out = {"classification_match": 0.0, "reconciliation_match": 0.0}
    want = ref.classify_and_reconcile(ref.SAMPLE_LOGS, ref.SAMPLE_WALL_TIME)
    try:
        got = analyze_log(ref.SAMPLE_LOGS, ref.SAMPLE_WALL_TIME)
    except Exception as e:
        out["_note"] = f"analyze_log raised {type(e).__name__}: {str(e)[:100]}"
        return out

    if isinstance(got, dict):
        if got.get("classes") == want["classes"]:
            out["classification_match"] = 1.0
        else:
            out["_note"] = f"classes got {got.get('classes')}, want {want['classes']}"
        if abs(got.get("ratio", 0) - want["ratio"]) < 1e-3:
            out["reconciliation_match"] = 1.0
        else:
            if "_note" not in out:
                out["_note"] = f"ratio got {got.get('ratio')}, want {want['ratio']}"
    else:
        out["_note"] = "analyze_log did not return a dict"
    return out
