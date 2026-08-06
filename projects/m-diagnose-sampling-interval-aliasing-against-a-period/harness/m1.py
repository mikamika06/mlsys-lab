import ref


def check(workdir):
    from sampler.core import simulate_execution, statistical_sample, ground_truth_fraction
    out = {"sampler_match": 0.0}
    ok = 0
    test_cases = [
        (1000, 10, 3, 5),
        (2000, 15, 5, 7),
        (1500, 20, 8, 11),
    ]
    for steps, period, duration, interval in test_cases:
        trace_ref = ref.simulate_execution(steps, period, duration)
        trace_got = simulate_execution(steps, period, duration)

        gt_ref = ref.ground_truth_fraction(trace_ref)
        gt_got = ground_truth_fraction(trace_got)

        samp_ref = ref.statistical_sample(trace_ref, interval)
        samp_got = statistical_sample(trace_got, interval)

        if abs(gt_ref - gt_got) < 1e-5 and abs(samp_ref - samp_got) < 1e-5:
            ok += 1
    out["sampler_match"] = float(ok)
    return out
