def select_2_4_mask(W: list[list[float]]) -> list[list[int]]:
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for start in range(0, cols, 4):
            group = W[r][start:start + 4]
            order = sorted(range(len(group)), key=lambda i: (-abs(float(group[i])), i))
            for idx in order[:2]:
                out[r][start + idx] = 1
    return out
