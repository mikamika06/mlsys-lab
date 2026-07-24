def choose_buckets(size_histogram, unit, cap, max_buckets):
    sizes = sorted(size_histogram)
    best = None

    def waste_for(chosen):
        total = 0
        for s, count in size_histogram.items():
            possible = [b for b in chosen if b >= s]
            if not possible:
                return None
            total += count * (min(possible) - s)
        return total

    def search(i, chosen, cost):
        nonlocal best
        if i == len(sizes):
            if len(chosen) <= max_buckets:
                w = waste_for(chosen)
                if w is not None:
                    candidate = (w, chosen)
                    if best is None or candidate[0] < best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                        best = candidate
            return
        search(i + 1, chosen, cost)
        if len(chosen) < max_buckets:
            new_cost = cost + sizes[i] * unit
            if new_cost <= cap:
                search(i + 1, chosen + (sizes[i],), new_cost)

    search(0, (), 0)
    return list(best[1]), best[0]
