import ref


def check(workdir):
    from recompile.guard import detect_recompile, enforce_guard
    out = {"guard_matched": 0.0}
    ok = 0
    for item in ref.GUARD_TESTS:
        want_detected = not item["history"] or item["shape"] not in item["history"]
        got_detected = detect_recompile(item["history"], item["shape"])
        try:
            enforce_guard(item["enabled"], got_detected)
            got_err = False
        except RuntimeError:
            got_err = True
        want_err = item["enabled"] and got_detected
        if got_detected == want_detected and got_err == want_err:
            ok += 1
    out["guard_matched"] = float(ok)
    return out
