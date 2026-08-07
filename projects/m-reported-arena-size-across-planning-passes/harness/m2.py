import ref


def check(workdir):
    from arena.separation import compute_separation_savings

    out = {"savings_match": 0.0, "methods_valid": 0.0}
    tensors = ref.TENSORS_LIST[0]
    want = ref.compute_separation_savings(tensors, "segmented", "isolated")
    got = compute_separation_savings(tensors, "segmented", "isolated")
    if got == want:
        out["savings_match"] = 1.0
    else:
        out["_note"] = f"got {got}, reference {want}"

    try:
        res = compute_separation_savings(tensors, "inline", "segmented")
        if isinstance(res, dict) and "savings_a" in res:
            out["methods_valid"] = 1.0
    except Exception as e:
        out["_note"] = f"methods raised error: {e}"

    return out
