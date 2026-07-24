def grade(sol, fx) -> dict:
    # Reference mapping implemented algorithmically.
    params = [False, False, False, True]
    grads  = [False, False, True,  True]
    optimizer = [False, True,  True,  True]

    ok = 1.0
    for stage in range(4):
        try:
            got = sol.zero_stage_sharding(stage)
            if not isinstance(got, tuple) or len(got) != 3:
                ok = 0.0
                break
            ref = (params[stage], grads[stage], optimizer[stage])
            if got != ref:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
