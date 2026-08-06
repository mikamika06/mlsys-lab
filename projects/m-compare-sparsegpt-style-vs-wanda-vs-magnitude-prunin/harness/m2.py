import ref


def check(workdir):
    from prune.wanda import wanda_prune

    W, X = ref.get_fixtures()
    try:
        res_normal = wanda_prune(W, X, 0.5, domain_shift=False)
        res_shifted = wanda_prune(W, X, 0.5, domain_shift=True)
    except Exception as e:
        return {"domain_debugged": 0.0, "norm_scaled": 0.0, "_note": f"raised {type(e).__name__}"}

    if "error" not in res_shifted or "norms" not in res_shifted:
        return {"domain_debugged": 0.0, "norm_scaled": 0.0, "_note": "missing keys"}

    if res_shifted["error"] <= res_normal["error"]:
        return {"domain_debugged": 0.0, "norm_scaled": 1.0, "_note": "shifted error should be higher"}

    return {"domain_debugged": 1.0, "norm_scaled": 1.0}
