def _reference(tree, accept_prob):
    def visit(node, path_prob):
        total = 0.0
        for child in tree[node]:
            child_prob = path_prob * float(accept_prob[child])
            total += child_prob
            total += visit(child, child_prob)
        return total

    return visit(0, 1.0)


def grade(sol, fx) -> dict:
    cases = [
        (
            [[1, 2], [3], [], []],
            [0.0, 0.5, 0.25, 0.8],
        ),
        (
            [[1], [2], [3], [4], []],
            [0.0, 0.9, 0.8, 0.7, 0.6],
        ),
        (
            [[1, 2, 3], [], [4], [5, 6], [], [], []],
            [0.0, 0.2, 0.4, 0.6, 0.5, 0.3, 0.9],
        ),
        (
            [list(range(1, 8)), [], [], [], [], [], [], []],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ),
    ]

    max_rel = 0.0
    for tree, probs in cases:
        try:
            got = float(sol.expected_accepted_length(tree, probs))
        except Exception:
            return {"rel_err": float("inf")}

        ref = _reference(tree, probs)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        max_rel = max(max_rel, err)

    return {"rel_err": max_rel}
