import ref


def check(workdir):
    from roofline.calc import roofline_speedup_ceiling, measure_speedup_error

    cases = ref.get_test_cases()
    max_err = 0.0
    for i, c in enumerate(cases):
        pred = roofline_speedup_ceiling(
            c["expected_intensity"],
            c["peak_flops"],
            c["peak_bandwidth"],
            c["baseline_time"],
            c["token_flops"],
            c["gamma"]
        )
        err = measure_speedup_error(pred, c["measured_speedup"])
        if err > max_err:
            max_err = err

    out = {"rel_err": float(max_err)}
    return out
