def bpe_merge(tokens, ranks):
    # TODO: chooses the first matching pair instead of the lowest-rank pair.
    # This fails when an earlier pair in the token list has lower priority.
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in ranks:
            selected = pair
            break
    else:
        return list(tokens)

    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == selected:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out
