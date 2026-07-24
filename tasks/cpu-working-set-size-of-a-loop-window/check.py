def _reference(addrs, w):
    from collections import defaultdict
    if not addrs or w <= 0:
        return 0
    n = len(addrs)
    window_size = min(w, n)
    freq = defaultdict(int)
    distinct = 0
    max_distinct = 0

    # initialise first window
    for i in range(window_size):
        a = addrs[i]
        if freq[a] == 0:
            distinct += 1
        freq[a] += 1
    max_distinct = distinct

    # slide the window
    for i in range(window_size, n):
        out = addrs[i - window_size]
        freq[out] -= 1
        if freq[out] == 0:
            distinct -= 1
        in_a = addrs[i]
        if freq[in_a] == 0:
            distinct += 1
        freq[in_a] += 1
        if distinct > max_distinct:
            max_distinct = distinct

    return int(max_distinct)

def grade(sol, fx) -> dict:
    cases = [
        ([], 5),
        ([1], 1),
        ([1,2,3,4,5], 2),
        ([1,2,1,3,2,4,5,6], 3),
        ([10]*1000 + list(range(200)), 150),
    ]
    ok = 1.0
    for addrs, w in cases:
        try:
            got = sol.max_working_set(addrs, w)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference(addrs, w)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
