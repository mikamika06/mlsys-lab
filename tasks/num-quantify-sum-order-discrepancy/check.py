def grade(sol, fx) -> dict:
    import numpy as np
    from mlsys.scorers import max_abs_err

    # Test arrays covering magnitude spread, signed zero and cancellation.
    cases = [
        np.array([1.0, 2.0, 3.0]),
        np.array([-1e308, 1e308, 1.0]),          # large cancellation
        np.array([0.0, -0.0, 1.0, -1.0]),         # signed zero
        np.array([1e-20]*1000 + [1e20])           # huge dynamic range
    ]

    errors = []
    for arr in cases:
        try:
            got = sol.sum_order_discrepancy(arr)
            if not isinstance(got, (tuple, list)) or len(got) != 3:
                return {"max_abs_err": float("inf")}
            s_asc, s_desc, s_pair = map(float, got)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref_asc = np.sum(np.sort(arr))
        ref_desc = np.sum(np.flip(np.sort(arr)))
        ref_pair = np.add.reduce(arr)

        err = max_abs_err(
            np.array([ref_asc, ref_desc, ref_pair]),
            np.array([s_asc, s_desc, s_pair])
        )
        errors.append(err)
    overall_error = max(errors) if errors else float("inf")
    return {"max_abs_err": overall_error}
