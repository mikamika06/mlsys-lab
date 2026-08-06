import ref


def check(workdir):
    from prune.compare import evaluate_pruning_methods

    W, X = ref.get_fixtures()
    try:
        res = evaluate_pruning_methods(W, X, 0.5)
    except Exception as e:
        return {"methods_compared": 0.0, "rel_err_valid": 0.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}

    if not isinstance(res, dict) or not all(k in res for k in ("magnitude", "wanda", "sparsegpt")):
        return {"methods_compared": 0.0, "rel_err_valid": 0.0, "_note": "missing keys in result dict"}

    if not all(isinstance(v, float) for v in res.values()):
        return {"methods_compared": 1.0, "rel_err_valid": 0.0, "_note": "values must be floats"}

    if not all(v >= 0.0 for v in res.values()):
        return {"methods_compared": 1.0, "rel_err_valid": 0.0, "_note": "errors must be non-negative"}

    return {"methods_compared": 1.0, "rel_err_valid": 1.0}
