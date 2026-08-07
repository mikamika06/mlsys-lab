def greedy_24_prune(W: list[list[float]]) -> list[float]:
    out = []
    for row in W:
        best = None
        for keep_0 in range(4):
            for keep_1 in range(keep_0 + 1, 4):
                dropped = 0.0
                for i in range(4):
                    if i != keep_0 and i != keep_1:
                        val = row[i]
                        if val < 0.0:
                            dropped += -val
                        else:
                            dropped += val
                if best is None or dropped < best:
                    best = dropped
        out.append(best)
    return out
