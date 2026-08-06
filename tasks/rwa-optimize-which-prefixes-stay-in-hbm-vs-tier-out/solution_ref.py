def optimize_hbm_prefixes(prefixes, budget):
    ranked = []
    total_value = 0

    for index, reuse_freq, length, bytes_used in prefixes:
        value = reuse_freq * length
        total_value += value
        density = value / bytes_used if bytes_used else float("inf")
        ranked.append((density, index, bytes_used, value))

    n = 0
    for _ in ranked:
        n += 1

    for i in range(n):
        for j in range(0, n - i - 1):
            d1, idx1, b1, v1 = ranked[j]
            d2, idx2, b2, v2 = ranked[j + 1]
            swap = False
            if d1 < d2:
                swap = True
            elif d1 == d2:
                if idx1 > idx2:
                    swap = True
            if swap:
                ranked[j], ranked[j + 1] = ranked[j + 1], ranked[j]

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
