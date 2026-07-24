def _oracle(prefixes, budget):
    ranked = []
    total_value = 0
    for index, reuse_freq, length, bytes_used in prefixes:
        value = reuse_freq * length
        total_value += value
        density = value / bytes_used if bytes_used else float("inf")
        ranked.append((density, index, bytes_used, value))

    ranked.sort(key=lambda x: (-x[0], x[1]))

    kept = []
    used = 0
    kept_value = 0
    for density, index, bytes_used, value in ranked:
        if used + bytes_used <= budget:
            kept.append(index)
            used += bytes_used
            kept_value += value

    hit_rate = kept_value / total_value if total_value else 0.0
    return kept, hit_rate


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                (10, 100, 20, 1000),
                (11, 20, 50, 100),
                (12, 10, 10, 500),
            ],
            600,
        ),
        (
            [
                (0, 50, 100, 1000),
                (1, 20, 100, 100),
                (2, 90, 10, 900),
                (3, 5, 1000, 200),
            ],
            1200,
        ),
        (
            [
                (5, 1, 1, 1),
                (6, 2, 10, 5),
                (7, 100, 1, 100),
                (8, 50, 5, 50),
            ],
            55,
        ),
        (
            [
                (20, 100, 30, 300),
                (21, 100, 20, 200),
                (22, 1, 1000, 10000),
            ],
            500,
        ),
    ]

    ok = 1.0
    for prefixes, budget in cases:
        ref = _oracle(prefixes, budget)
        try:
            got = sol.optimize_hbm_prefixes(list(prefixes), budget)
            got = (list(got[0]), float(got[1]))
        except Exception:
            ok = 0.0
            break

        if got[0] != ref[0] or got[1] != ref[1]:
            ok = 0.0
            break

    return {"exact_match": ok}
