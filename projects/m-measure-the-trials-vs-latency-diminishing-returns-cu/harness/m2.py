import ref


def check(workdir):
    from metatune.measure import measure_diminishing_returns

    trials_list = [1, 5, 10, 20, 50, 100]
    base_latency = 10.0
    min_latency = 2.0
    expected = ref.compute_reference_curve(trials_list, base_latency, min_latency)

    try:
        got = measure_diminishing_returns(trials_list, base_latency, min_latency)
    except Exception as e:
        return {"curve_matched": 0.0, "_note": f"measure_diminishing_returns raised {type(e).__name__}"}

    if not isinstance(got, list) or len(got) != len(expected):
        return {"curve_matched": 0.0, "_note": "curve length mismatch"}

    match = 1.0
    for g, e in zip(got, expected):
        if abs(float(g) - float(e)) > 1e-4:
            match = 0.0
            break

    return {"curve_matched": match}
