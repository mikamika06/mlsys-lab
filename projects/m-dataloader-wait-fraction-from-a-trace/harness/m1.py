import ref


def check(workdir):
    from pipeline.trace import compute_wait_fraction
    data = ref.generate_test_data()
    max_rel_err = 0.0
    for trace in data["traces"]:
        want = ref.compute_wait_fraction(trace)
        got = compute_wait_fraction(trace)
        rel_err = abs(got - want) / (abs(want) + 1e-9)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"wait_fraction_rel_err": float(max_rel_err)}
