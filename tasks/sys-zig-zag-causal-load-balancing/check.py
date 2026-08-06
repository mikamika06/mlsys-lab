def _oracle(num_ranks):
    total = 2 * num_ranks
    out = [-1] * total
    for r in range(num_ranks):
        out[r] = r
        out[total - 1 - r] = r
    return out


def _loads(assignment, num_ranks):
    loads = [0] * num_ranks
    for i, r in enumerate(assignment):
        loads[r] += i + 1
    return loads


def _as_rank_list(got, num_ranks):
    out = []
    for r in got:
        value = int(r)
        if value != r or value < 0 or value >= num_ranks:
            return None
        out.append(value)
    return out


def grade(sol, fx) -> dict:
    cases = [1, 2, 3, 4, 5, 8, 13]
    bad = {"exact_match": 0.0, "imbalance": float("inf")}

    exact = 1.0
    worst_imbalance = 1.0

    for num_ranks in cases:
        try:
            got = list(sol.zigzag_assignment(num_ranks))
        except Exception:
            return bad

        ref = _oracle(num_ranks)
        if len(got) != len(ref):
            return bad

        try:
            ranks = _as_rank_list(got, num_ranks)
        except (TypeError, ValueError):
            return bad
        if ranks is None:
            return bad

        if ranks != ref:
            exact = 0.0

        loads = _loads(ranks, num_ranks)
        if min(loads) <= 0:
            return bad
        worst_imbalance = max(worst_imbalance, max(loads) / min(loads))

    return {"exact_match": exact, "imbalance": float(worst_imbalance)}
