import ref

def check(workdir):
    from offload.model_size import max_trainable_model_size
    out = {"size_matched": 0.0, "cases": float(len(ref.SCENARIOS))}
    ok = 0
    for i, s in enumerate(ref.SCENARIOS):
        want = ref.max_trainable_model_size(**s)
        got = max_trainable_model_size(**s)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got {got}, reference {want}"
    out["size_matched"] = float(ok)
    return out
