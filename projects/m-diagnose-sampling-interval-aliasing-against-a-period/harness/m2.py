import ref


def check(workdir):
    from sampler.diagnose import compute_bias
    out = {"aliasing_match": 0.0}
    ok = 0
    test_cases = [
        (100, 20, [10, 20, 21], 5000),
        (200, 30, [15, 30, 31], 5000),
        (150, 25, [25, 50, 53], 5000),
    ]
    for period, duration, intervals, steps in test_cases:
        ref_res = ref.compute_bias(period, duration, intervals, steps=steps)
        got_res = compute_bias(period, duration, intervals, steps=steps)

        match = True
        if len(ref_res) != len(got_res):
            match = False
        else:
            for r, g in zip(ref_res, got_res):
                if abs(r["bias"] - g["bias"]) > 1e-4:
                    match = False
                    break
        if match:
            ok += 1
    out["aliasing_match"] = float(ok)
    return out
