def tree_reduce(values):
    current = list(values)
    trace = []

    if not current:
        return 0.0, trace

    while len(current) > 1:
        nxt = []
        i = 0
        while i + 1 < len(current):
            nxt.append(current[i] + current[i + 1])
            i += 2
        if i < len(current):
            nxt.append(current[i])
        trace.append([len(nxt)])
        current = nxt

    return current[0], trace
