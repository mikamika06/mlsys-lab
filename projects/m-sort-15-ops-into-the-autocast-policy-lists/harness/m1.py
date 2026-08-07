import ref


def check(workdir):
    from ampcheck import ops

    out = {"ops_matched": 0.0}
    try:
        got = ops.classify_ops()
        want = ref.classify_ops()
        if got == want:
            out["ops_matched"] = 1.0
        else:
            out["_note"] = f"got policy {got}, reference {want}"
    except Exception as e:
        out["_note"] = f"error in classify_ops: {str(e)[:120]}"
    return out
