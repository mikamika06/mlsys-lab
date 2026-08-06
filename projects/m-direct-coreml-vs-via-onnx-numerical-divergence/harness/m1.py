import ref

def check(workdir):
    from coreml_audit import divergence
    out = {"max_abs_err_matched": 0.0, "configs": float(len(ref.MODELS))}
    ok = 0
    for i, m in enumerate(ref.MODELS):
        want = ref.compute_max_abs_err(m)
        got = divergence.max_abs_error(m["direct"], m["onnx"])
        if abs(want - got) < 1e-6:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: got {got}, reference {want}"
    out["max_abs_err_matched"] = float(ok)
    return out
