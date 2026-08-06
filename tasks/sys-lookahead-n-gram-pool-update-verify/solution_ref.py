def manual_sort(items, key_fn):
    arr = list(items)
    for i in range(1, len(arr)):
        curr = arr[i]
        curr_key = key_fn(curr)
        j = i - 1
        while j >= 0 and key_fn(arr[j]) > curr_key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = curr
    return arr


def lookahead_pool_update_verify(trace, n, lookahead, pool_size):
    counts = {}

    def add_ngram(pos):
        if pos >= n - 1:
            ctx = tuple(trace[pos - n + 1:pos])
            key = (ctx, trace[pos])
            counts[key] = counts.get(key, 0) + 1

    for i in range(len(trace)):
        add_ngram(i)

    verified = []
    i = n - 1

    while i < len(trace):
        current = list(trace[:i])
        proposals = []

        for _ in range(lookahead):
            if len(current) < n - 1:
                break
            ctx = tuple(current[-(n - 1):])
            choices = []
            for (c, t), freq in counts.items():
                if c == ctx:
                    choices.append((freq, t))
            if not choices:
                break
            sorted_choices = manual_sort(choices, lambda x: (-x[0], x[1]))
            current.append(sorted_choices[0][1])
            proposals.append(sorted_choices[0][1])

        matched = 0
        for off, token in enumerate(proposals):
            if i + off < len(trace) and trace[i + off] == token:
                verified.append(token)
                matched += 1
            else:
                break

        m = matched if matched > 1 else 1
        limit = i + m
        if len(trace) < limit:
            limit = len(trace)

        for j in range(i, limit):
            add_ngram(j)

        i += m

    entries = list(counts.items())
    sorted_entries = manual_sort(entries, lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return verified, [key for key, _ in sorted_entries[:pool_size]]
