import ref


def check(workdir):
    from edgeml.oom import simulate_oom
    out = {"oom_points_matched": 0.0}
    ok = 0
    for i, case in enumerate(ref.OOM_TEST_CASES):
        want = ref.simulate_oom(case)
        got = simulate_oom(case)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    if ok == len(ref.OOM_TEST_CASES):
        out["oom_points_matched"] = 1.0
    return out
