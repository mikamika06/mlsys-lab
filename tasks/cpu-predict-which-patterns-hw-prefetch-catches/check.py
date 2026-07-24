def grade(sol, fx) -> dict:
    # Reference: classify each pattern as caught (True) or not (False)
    # by a typical stream+stride hardware prefetcher.
    #
    # Pattern 0: sequential stride=4B        -> True  (stream prefetch)
    # Pattern 1: fixed stride=16B            -> True  (stride prefetch)
    # Pattern 2: random                      -> False (no pattern)
    # Pattern 3: pointer chase               -> False (addr depends on data)
    # Pattern 4: stride=4096B (page size)    -> False (crosses page per step)
    ref = [True, True, False, False, False]

    try:
        result = list(sol.classify_prefetch())
    except Exception:
        return {"exact_match": 0.0}

    if len(result) != 5:
        return {"exact_match": 0.0}

    matches = sum(1 for r, e in zip(result, ref) if bool(r) == bool(e))
    exact_match = 1.0 if matches == 5 else 0.0
    return {"exact_match": exact_match}
