def _oracle(lengths, page_size):
    total = sum(lengths)
    paged_total = 0
    for length in lengths:
        pages = (length + page_size - 1) // page_size
        paged_total += pages * page_size
    pre_total = len(lengths) * max(lengths)
    return (
        (paged_total - total) / paged_total,
        (pre_total - total) / pre_total,
    )


def grade(sol, fx) -> dict:
    cases = [
        ([10, 17, 33], 16),
        ([1, 2, 3, 4, 5], 8),
        ([64, 64, 64], 16),
        ([7, 31, 32, 65, 90], 32),
        ([100, 101, 102, 103, 104], 64),
    ]

    max_err = 0.0
    for lengths, page_size in cases:
        try:
            got = sol.fragmentation_waste_fraction(list(lengths), page_size)
            if len(got) != 2:
                return {"size_ratio": 1.0}
            ref = _oracle(list(lengths), page_size)
            err = max(abs(float(got[0]) - ref[0]), abs(float(got[1]) - ref[1]))
            max_err = max(max_err, err)
        except Exception:
            return {"size_ratio": 1.0}

    return {"size_ratio": max_err}
