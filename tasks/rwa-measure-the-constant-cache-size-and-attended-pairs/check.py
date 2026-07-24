def _oracle(k, w, n):
    cache = [min(k + w, i) for i in range(1, n + 1)]
    pairs = [min(w, i - 1) for i in range(1, n + 1)]
    return cache, pairs

def grade(sol, fx) -> dict:
    # Test cases with varying hyper‑parameters
    cases = [
        (2, 3, 5),
        (0, 0, 10),
        (5, 7, 20),
        (1, 100, 50)
    ]
    ok = 1.0
    for k, w, n in cases:
        try:
            got = sol.measure_cache_and_attended(k, w, n)
            if not isinstance(got, (tuple, list)) or len(got) != 2:
                ok = 0.0
                break
            cache, pairs = got
            # Convert to plain lists for comparison
            cache = list(cache)
            pairs = list(pairs)
        except Exception:
            ok = 0.0
            break

        ref_cache, ref_pairs = _oracle(k, w, n)

        if cache != ref_cache or pairs != ref_pairs:
            ok = 0.0
            break
    return {"exact_match": ok}
