def find_reusable_chunks(trace, chunk_size):
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
