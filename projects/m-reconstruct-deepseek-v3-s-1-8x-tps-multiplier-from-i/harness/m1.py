import ref

def check(workdir):
    from mtpcalc.multiplier import compute_tps_multiplier
    test_rates = [0.5, 0.75, 0.85, 0.9]
    max_rel_err = 0.0
    for rate in test_rates:
        want = ref.compute_tps_multiplier(rate)
        try:
            got = float(compute_tps_multiplier(rate))
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"raised {type(e).__name__}"}
        err = abs(got - want) / (abs(want) + 1e-9)
        if err > max_rel_err:
            max_rel_err = err
    return {"rel_err": float(max_rel_err)}
