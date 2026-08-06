import math

def fuse_mask_scale_softmax(keys: list[list[float]], values: list[list[float]], mask: list[list[float]], scale: float) -> list[list[float]]:
    N = len(keys)
    D = len(keys[0])
    M = len(values)
    out = [[0.0] * M for _ in range(N)]
    for i in range(N):
        row = [0.0] * M
        for j in range(M):
            s = 0.0
            for k in range(D):
                s += keys[i][k] * values[j][k]
            row[j] = s * scale + mask[i][j]

        max_val = row[0]
        for j in range(1, M):
            if row[j] > max_val:
                max_val = row[j]

        exp_row = [0.0] * M
        exp_sum = 0.0
        for j in range(M):
            val = math.exp(row[j] - max_val)
            exp_row[j] = val
            exp_sum += val

        for j in range(M):
            out[i][j] = exp_row[j] / exp_sum

    return out
