def max_working_set(addrs, window_size):
    from collections import defaultdict

    n = len(addrs)
    if not addrs or window_size <= 0:
        return 0

    w = min(window_size, n)
    freq = defaultdict(int)
    distinct = 0
    max_distinct = 0

    # build first window
    for i in range(w):
        a = addrs[i]
        if freq[a] == 0:
            distinct += 1
        freq[a] += 1
    max_distinct = distinct

    # slide over remaining elements
    for i in range(w, n):
        out_addr = addrs[i - w]
        freq[out_addr] -= 1
        if freq[out_addr] == 0:
            distinct -= 1

        in_addr = addrs[i]
        if freq[in_addr] == 0:
            distinct += 1
        freq[in_addr] += 1

        if distinct > max_distinct:
            max_distinct = distinct

    return int(max_distinct)
