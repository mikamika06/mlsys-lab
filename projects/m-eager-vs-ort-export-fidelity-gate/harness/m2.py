import ref


def check(workdir):
    from exportgate.tracebacks import classify_traceback

    out = {"classification_matched": 0.0, "tests": float(len(ref.TRACEBACK_TESTS))}
    ok = 0
    for i, (tb_str, expected) in enumerate(ref.TRACEBACK_TESTS):
        try:
            got = classify_traceback(tb_str)
            if got == expected:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"test {i}: got {got}, expected {expected}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"test {i} raised {type(e).__name__}: {str(e)[:100]}"
    out["classification_matched"] = float(ok)
    return out
