import ref


def check(workdir):
    from profiler.schedule import get_step_action

    out = {"states_matched": 0.0}
    ok = 0
    for i, tc in enumerate(ref.SCHEDULE_TEST_CASES):
        params = {k: tc[k] for k in ("skip_first", "wait", "warmup", "active", "repeat")}
        matched = True
        for step in tc["steps"]:
            want = ref.ref_get_step_action(step, **params)
            got = get_step_action(step, **params)
            if got != want:
                matched = False
                if "_note" not in out:
                    out["_note"] = f"case {i} step {step}: got '{got}', want '{want}'"
                break
        if matched:
            ok += 1

    if ok == len(ref.SCHEDULE_TEST_CASES):
        out["states_matched"] = 1.0
    return out
