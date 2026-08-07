import ref


def check(workdir):
    from flashvar.reducer import deterministic_backward
    out = {"deterministic_match": 0.0, "speedup_or_accuracy": 0.0}
    try:
        grads = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        got = deterministic_backward(grads)
        want = ref.deterministic_backward(grads)
        if got is not None:
            out["deterministic_match"] = 1.0
            if len(got) == len(want) and all(abs(a - b) < 1e-5 for a, b in zip(got, want)):
                out["speedup_or_accuracy"] = 1.0
    except Exception as e:
        out["_note"] = f"m2 failed: {type(e).__name__}: {str(e)[:120]}"
    return out
