import ref


def check(workdir):
    from int8opt.canonicalize import canonicalize_qdq

    out = {"graph_canonicalized": 0.0}
    try:
        mock = {"nodes": ["QuantizeLinear", "MatMul", "DequantizeLinear"]}
        got = canonicalize_qdq(mock)
        want = ref.canonicalize_qdq(mock)
        if got == want:
            out["graph_canonicalized"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error during execution: {str(e)[:120]}"
    return out
