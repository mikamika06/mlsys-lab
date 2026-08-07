import math


def _hadamard(n: int) -> list[list[float]]:
    h = [[1.0]]
    while len(h) < n:
        new_h = []
        for row in h:
            new_h.append(row + row)
        for row in h:
            new_h.append(row + [-val for val in row])
        h = new_h
    scale = math.sqrt(n)
    h_out = []
    for i in range(len(h)):
        out_row = []
        for j in range(len(h[0])):
            out_row.append(h[i][j] / scale)
        h_out.append(out_row)
    return h_out


def _ratio(X: list[list[float]]) -> list[float]:
    rows = len(X)
    cols = len(X[0])
    res = []
    for i in range(rows):
        sum_sq = 0.0
        max_abs = 0.0
        for j in range(cols):
            val = X[i][j]
            sum_sq += val * val
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val
        rms = math.sqrt(sum_sq / cols)
        res.append(max_abs / rms)
    return res


def outlier_ratio_before_after_rotation(X: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-token peak/rms ratio (over channels) before and after rotating
    the batch with a normalized Sylvester-Hadamard matrix, X_rot = X @ H^T."""
    d = len(X[0])
    H = _hadamard(d)

    rows = len(X)
    H_rows = len(H)
    H_cols = len(H[0])

    Xrot = []
    for i in range(rows):
        rot_row = []
        for j in range(H_rows):
            acc = 0.0
            for k in range(d):
                acc += X[i][k] * H[j][k]
            rot_row.append(acc)
        Xrot.append(rot_row)

    return _ratio(X), _ratio(Xrot)
