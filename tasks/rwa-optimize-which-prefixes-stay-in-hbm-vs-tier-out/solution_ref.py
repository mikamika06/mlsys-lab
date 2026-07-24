def optimize_hbm_prefixes(prefixes, budget):
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
