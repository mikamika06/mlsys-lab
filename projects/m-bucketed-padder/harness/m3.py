import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_unaligned_ladder": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:  # noqa: BLE001
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import padder.ladder as l_mod
    good_fn = l_mod.find_optimal_ladder

    def broken_find_optimal_ladder(lengths, candidate_bounds, max_buckets, compilation_cost, alignment=1):
        candidates = sorted(list({b for b in candidate_bounds if b >= max(lengths)}))
        import itertools
        from padder.cost import compute_padding_waste
        best_cost = float("inf")
        best_ladder = []
        for k in range(1, max_buckets + 1):
            for combo in itertools.combinations(candidates, k):
                ladder = list(combo)
                if max(ladder) < max(lengths):
                    continue
                waste, _ = compute_padding_waste(lengths, ladder)
                cost = waste + len(ladder) * compilation_cost
                if cost < best_cost:
                    best_cost = cost
                    best_ladder = sorted(ladder)
        return best_ladder, best_cost

    l_mod.find_optimal_ladder = broken_find_optimal_ladder
    import padder
    padder.ladder.find_optimal_ladder = broken_find_optimal_ladder

    try:
        out["catches_unaligned_ladder"] = 0.0 if _survives(path) else 1.0
    finally:
        l_mod.find_optimal_ladder = good_fn
        padder.ladder.find_optimal_ladder = good_fn

    return out
