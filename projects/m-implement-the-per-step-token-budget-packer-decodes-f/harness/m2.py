import ref


def check(workdir):
    from tokenpacker.steps import compute_steps

    out = {"steps_matched": 0.0}
    ok = True
    test_cases = [(1024, 512, 64), (256, 128, 16), (2048, 1024, 128)]
    for L, b, d in test_cases:
        want = ref.compute_steps(L, b, d)
        got = compute_steps(L, b, d)
        if got != want:
            ok = False
            break
    out["steps_matched"] = 1.0 if ok else 0.0
    return out
