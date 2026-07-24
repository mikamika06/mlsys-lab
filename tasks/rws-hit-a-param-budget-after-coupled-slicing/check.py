import numpy as np


def _total_params(coefs: np.ndarray, consts: np.ndarray, d: int) -> int:
    dim0 = coefs[:, 0] * d + consts[:, 0]
    dim1 = coefs[:, 1] * d + consts[:, 1]
    return int(np.sum(dim0 * dim1))


def _oracle(coefs, consts, widths, budget):
    coefs = np.asarray(coefs, dtype=np.int64)
    consts = np.asarray(consts, dtype=np.int64)
    widths = np.asarray(widths, dtype=np.int64)
    budget = int(budget)

    counts = np.array([_total_params(coefs, consts, int(d)) for d in widths])
    feasible = widths[counts <= budget]
    if feasible.size > 0:
        chosen = int(feasible.max())
    else:
        chosen = int(widths[int(np.argmin(counts))])
    return chosen, _total_params(coefs, consts, chosen)


def _cases():
    cases = []

    # hand-checkable: two tensors (3d,d) and (d,1) -> P(d) = 3d^2 + d
    coefs0 = np.array([[3, 1], [1, 0]], dtype=np.int64)
    consts0 = np.array([[0, 0], [0, 1]], dtype=np.int64)
    widths0 = np.array([2, 3, 4], dtype=np.int64)
    cases.append((coefs0, consts0, widths0, 40))

    # budget below every candidate -> fallback to smallest param count
    coefs1 = np.array([[1, 1]], dtype=np.int64)
    consts1 = np.array([[0, 0]], dtype=np.int64)
    widths1 = np.array([10, 20, 30], dtype=np.int64)
    cases.append((coefs1, consts1, widths1, 1))

    # random synthetic architecture, unsorted widths
    rng = np.random.default_rng(2)
    T = 12
    coefs2 = rng.integers(0, 5, size=(T, 2)).astype(np.int64)
    consts2 = rng.integers(0, 50, size=(T, 2)).astype(np.int64)
    widths2 = rng.permutation(np.arange(8, 128, 8)).astype(np.int64)
    counts2 = [_total_params(coefs2, consts2, int(d)) for d in np.sort(widths2)]
    mid_budget = int(np.median(counts2))
    cases.append((coefs2, consts2, widths2, mid_budget))

    return cases


def grade(sol, fx) -> dict:
    all_cases = [(
        fx["cs_coefs"], fx["cs_consts"], fx["cs_widths"], int(fx["cs_budget"])
    )] + _cases()

    hits = 0
    for coefs, consts, widths, budget in all_cases:
        exp_width, exp_count = _oracle(coefs, consts, widths, budget)
        try:
            got_width, got_count = sol.pick_width_for_budget(
                np.array(coefs, copy=True), np.array(consts, copy=True),
                np.array(widths, copy=True), budget,
            )
            got_width = int(got_width)
            got_count = int(got_count)
        except Exception:
            continue

        if got_width == exp_width and got_count == exp_count:
            hits += 1

    return {"exact_match": hits / len(all_cases)}
