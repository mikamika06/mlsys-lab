def _total_params(coefs: list[list[int]], consts: list[list[int]], d: int) -> int:
    total = 0
    num_rows = len(coefs)
    for i in range(num_rows):
        dim0 = int(coefs[i][0]) * d + int(consts[i][0])
        dim1 = int(coefs[i][1]) * d + int(consts[i][1])
        total += dim0 * dim1
    return int(total)


def pick_width_for_budget(coefs: list[list[int]], consts: list[list[int]], widths: list[int], budget: int) -> tuple[int, int]:
    """Sweep every candidate width, compute the exact coupled parameter
    count P(d) = sum_t shape_t(d)[0] * shape_t(d)[1], and return the
    largest width whose P(d) fits `budget` (falling back to the width with
    the smallest P(d) if none fits), together with its exact param count.
    """
    budget = int(budget)

    num_widths = len(widths)
    counts_list = []
    for i in range(num_widths):
        d = int(widths[i])
        counts_list.append(_total_params(coefs, consts, d))

    best_feasible_width = None
    min_count = None
    min_count_width = None

    for i in range(num_widths):
        w = int(widths[i])
        c = counts_list[i]

        if min_count is None or c < min_count:
            min_count = c
            min_count_width = w

        if c <= budget:
            if best_feasible_width is None or w > best_feasible_width:
                best_feasible_width = w

    if best_feasible_width is not None:
        chosen = best_feasible_width
    else:
        chosen = min_count_width

    return int(chosen), _total_params(coefs, consts, int(chosen))
