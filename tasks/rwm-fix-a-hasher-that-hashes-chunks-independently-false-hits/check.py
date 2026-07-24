def _oracle(trace, chunk_size):
    base = 257
    mod = 2**61 - 1
    states = [0]
    for value in trace:
        states.append((states[-1] * base + value) % mod)

    seen = {}
    reusable = []
    for i in range(len(trace) - chunk_size + 1):
        key = (states[i], tuple(trace[i:i + chunk_size]))
        if key in seen:
            reusable.append(i)
        else:
            seen[key] = i
    return reusable


def grade(sol, fx) -> dict:
    cases = [
        ([4, 9, 2, 7, 9, 2, 7], 3),
        ([1, 2, 3, 1, 2, 3, 1, 2, 3], 3),
        ([5, 8, 5, 8, 5, 8, 5, 8], 2),
        ([10, 20, 30, 10, 20, 30, 40, 10, 20, 30], 3),
        (list(range(20)) + list(range(10, 20)), 4),
    ]

    ok = 1.0
    for trace, chunk_size in cases:
        expected = _oracle(trace, chunk_size)
        try:
            got = sol.find_reusable_chunks(list(trace), chunk_size)
        except Exception:
            ok = 0.0
            break
        if list(got) != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
