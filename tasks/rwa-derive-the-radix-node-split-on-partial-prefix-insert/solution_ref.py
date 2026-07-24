def derive_split(existing, incoming):
    limit = min(len(existing), len(incoming))
    for i in range(limit):
        if existing[i] != incoming[i]:
            return i, 2
    return -1, 1
