import ref


def check(workdir):
    from profiler_utils.schedule import active_windows_at_step
    test_cases = [
        (10, 1, 1, 2, 0),
        (25, 2, 2, 4, 3),
        (5, 0, 0, 2, 0),
        (50, 5, 5, 10, 0)
    ]
    ok = 0
    for step, w, wu, act, rep in test_cases:
        want = ref.active_windows_at_step(step, w, wu, act, rep)
        try:
            got = active_windows_at_step(step, w, wu, act, rep)
            if got == want:
                ok += 1
        except Exception:
            pass
    out = {"schedule_match": 1.0 if ok == len(test_cases) else 0.0}
    if ok != len(test_cases):
        out["_note"] = f"passed {ok}/{len(test_cases)} schedule test cases"
    return out
