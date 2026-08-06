import ref


def check(workdir):
    from fa_tradeoff.precision import compute_relative_error
    q, k, v, mask = ref.generate_inputs()
    got = compute_relative_error(q)
    want = ref.reference_precision_error(q)
    err = abs(got - want)
    passed = 1.0 if err < 1e-5 else 0.0
    return {"rel_err_match": passed, "_note": f"got {got}, want {want}"}
