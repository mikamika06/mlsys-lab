import ref


def check(workdir):
    from blockinf.prune import select_layers_to_remove
    bi_scores = [0.4, 0.1, 0.8, 0.2]
    num_remove = 2
    want = ref.select_layers_to_remove(bi_scores, num_remove)
    try:
        got = select_layers_to_remove(bi_scores, num_remove)
    except Exception as e:
        return {"selection_match": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    match = 1.0 if sorted(got) == sorted(want) else 0.0
    out = {"selection_match": match}
    if match == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
