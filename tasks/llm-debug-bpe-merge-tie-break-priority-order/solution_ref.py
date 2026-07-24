def bpe_merge(tokens, ranks):
    pairs = []
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in ranks:
            pairs.append(pair)

    if not pairs:
        return list(tokens)

    best = min(pairs, key=lambda p: ranks[p])

    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out
