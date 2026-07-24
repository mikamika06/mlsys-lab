def _oracle(size_histogram, unit, cap, max_buckets):
    sizes = sorted(size_histogram)
    best = None

    def waste_for(chosen):
        total = 0
        for s, count in size_histogram.items():
            bigger = [b for b in chosen if b >= s]
            if not bigger:
                return None
            total += count * (min(bigger) - s)
        return total

    dp = {(0, 0): [()]}
    for size in sizes:
        next_dp = dict(dp)
        cost = size * unit
        for (used_cost, used_count), sets in dp.items():
            if used_count < max_buckets and used_cost + cost <= cap:
                key = (used_cost + cost, used_count + 1)
                if key not in next_dp:
                    next_dp[key] = []
                for chosen in sets:
                    next_dp[key].append(chosen + (size,))
        dp = next_dp

    for sets in dp.values():
        for chosen in sets:
            w = waste_for(chosen)
            if w is None:
                continue
            candidate = (w, list(chosen))
            if best is None or candidate[0] < best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate

    return (best[1], best[0])


def grade(sol, fx) -> dict:
    cases = [
        ({3: 10, 5: 4, 9: 2}, 2, 20, 2),
        ({2: 8, 4: 5, 7: 3, 11: 1}, 3, 48, 3),
        ({1: 20, 6: 2, 10: 2}, 4, 56, 2),
        ({5: 7, 8: 6, 12: 1, 15: 1}, 2, 60, 3),
        ({3: 1, 4: 10, 6: 4, 9: 2, 14: 1}, 1, 35, 3),
    ]
    ok = 1.0
    for hist, unit, cap, k in cases:
        try:
            got = sol.choose_buckets(dict(hist), unit, cap, k)
            got = (list(got[0]), int(got[1]))
        except Exception:
            ok = 0.0
            break
        ref = _oracle(hist, unit, cap, k)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
