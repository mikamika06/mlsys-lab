def select_variant_set(variants, download_budget):
    valid_variants = [
        v for v in variants if v.get("download_bytes", 0) <= download_budget
    ]
    best_combo = []
    best_utility = -1.0

    n = len(valid_variants)
    for i in range(1 << n):
        subset = [valid_variants[j] for j in range(n) if (i & (1 << j))]
        total_size = sum(v["download_bytes"] for v in subset)
        if total_size <= download_budget:
            total_utility = sum(v["utility"] for v in subset)
            if total_utility > best_utility:
                best_utility = total_utility
                best_combo = subset
            elif total_utility == best_utility:
                if total_size < sum(v["download_bytes"] for v in best_combo):
                    best_combo = subset

    return sorted([v["id"] for v in best_combo])
