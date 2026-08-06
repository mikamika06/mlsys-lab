import ref


def check(workdir):
    from runner_limits.thrash import detect_swap_thrash

    out = {"thrash_matched": 0.0}
    test_streams = [
        ([{"tok_s": 2.0, "memory_pressure": 0.9, "swap_active": True}], True),
        ([{"tok_s": 20.0, "memory_pressure": 0.3, "swap_active": False}], False),
        ([{"tok_s": 3.0, "memory_pressure": 0.88, "swap_active": False}], True),
    ]
    ok = True
    for i, (stream, expected) in enumerate(test_streams):
        try:
            res = detect_swap_thrash(stream)
            if res != expected:
                ok = False
                out["_note"] = f"stream {i}: expected {expected}, got {res}"
                break
        except Exception as e:
            ok = False
            out["_note"] = f"stream {i} raised {type(e).__name__}: {e}"
            break
    if ok:
        out["thrash_matched"] = 1.0
    return out
