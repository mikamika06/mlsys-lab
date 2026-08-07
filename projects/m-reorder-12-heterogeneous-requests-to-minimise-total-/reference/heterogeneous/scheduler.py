def reorder_requests(requests):
    reqs = list(requests)
    if not reqs:
        return []
    def cost(r1, r2):
        diffs = sum(1 for k in r1 if r1.get(k) != r2.get(k))
        return diffs
    current = min(reqs, key=lambda r: str(r))
    remaining = [r for r in reqs if r != current]
    ordered = [current]
    while remaining:
        nxt = min(remaining, key=lambda r: (cost(ordered[-1], r), str(r)))
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered
