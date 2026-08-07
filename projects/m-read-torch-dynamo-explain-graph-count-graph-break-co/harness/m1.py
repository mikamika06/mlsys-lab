import ref

def check(workdir):
    from dynamotrace.analyzer import analyze_function

    args = ref.get_sample_args()
    try:
        got = analyze_function(ref.sample_target_fn, args)
    except Exception as e:
        return {"explain_match": 0.0, "_note": f"execution raised error: {type(e).__name__}: {str(e)[:120]}"}

    import torch
    explanation = torch._dynamo.explain(ref.sample_target_fn)(*args)
    want_graphs = int(explanation.graph_count)
    want_breaks = int(explanation.graph_break_count)

    match = 1.0 if got.get("graph_count") == want_graphs and got.get("break_count") == want_breaks else 0.0
    out = {"explain_match": match}
    if match == 0.0:
        out["_note"] = f"got {got}, expected graph_count={want_graphs}, break_count={want_breaks}"
    return out
