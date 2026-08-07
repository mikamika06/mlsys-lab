def reorder_requests(requests):
    def cost(r1, r2):
        reloads = 0
        if r1.get("model") != r2.get("model"):
            reloads += 2
        if r1.get("num_ctx") != r2.get("num_ctx"):
            reloads += 1
        return reloads

    best_order = None
    min_cost = float("inf")

    import itertools
    for p in itertools.permutations(requests):
        c = 0
        for i in range(len(p) - 1):
            c += cost(p[i], p[i+1])
        if c < min_cost:
            min_cost = c
            best_order = list(p)
    return best_order
