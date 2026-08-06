import ref


def check(workdir):
    from benchrunner.selector import select_runner

    out = {"selections_matched": 0.0}
    ok = 0
    for i, case in enumerate(ref.SELECTOR_CASES):
        want = ref.select_runner(case)
        got = select_runner(case)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    if ok == len(ref.SELECTOR_CASES):
        out["selections_matched"] = 1.0
    return out
