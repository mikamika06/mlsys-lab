def grade(sol, fx) -> dict:
    # Reference ranking from lowest to highest MLP:
    # pointer_chase (MLP=1, serial) < sequential (HW prefetch, moderate MLP)
    # < strided (multiple streams) < scatter_gather (many independent misses)
    ref = ["pointer_chase", "sequential", "strided", "scatter_gather"]

    try:
        result = list(sol.rank_by_mlp())
    except Exception:
        return {"exact_match": 0.0}

    if len(result) != 4:
        return {"exact_match": 0.0}

    exact_match = 1.0 if list(result) == ref else 0.0
    return {"exact_match": exact_match}
