def _reference_blelloch(values):
    a = list(values)
    n = len(a)

    stride = 1
    while stride < n:
        step = stride * 2
        for i in range(0, n, step):
            a[i + step - 1] += a[i + stride - 1]
        stride = step

    a[-1] = 0

    stride = n // 2
    while stride >= 1:
        step = stride * 2
        for i in range(0, n, step):
            left = i + stride - 1
            right = i + step - 1
            t = a[left]
            a[left] = a[right]
            a[right] += t
        stride //= 2

    return a


def grade(sol, fx) -> dict:
    cases = [
        [3, 1, 7, 0, 4, 1, 6, 3],
        [5, -2, 8, 1],
        [9, 0, -4, 6, 2, 3, 1, -1],
        list(range(16)),
        [7, 7],
    ]

    ok = 1.0
    for values in cases:
        original = list(values)
        try:
            got = sol.blelloch_scan(values)
        except Exception:
            ok = 0.0
            break

        if values != original:
            ok = 0.0
            break

        if list(got) != _reference_blelloch(values):
            ok = 0.0
            break

    return {"exact_match": ok}
