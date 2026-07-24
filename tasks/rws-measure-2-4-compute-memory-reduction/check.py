import numpy as np


def _oracle(W):
    W = np.asarray(W, dtype=np.float16)
    nnz = int(np.count_nonzero(W))
    total = int(W.size)
    groups = total // 4
    dense_bytes = int(W.nbytes)
    packed_bytes = int(nnz * np.dtype(np.float16).itemsize + ((2 * groups + 7) // 8))
    density = float(nnz / total)
    size_ratio = float(dense_bytes / packed_bytes)
    return density, packed_bytes, size_ratio


def grade(sol, fx) -> dict:
    cases = []
    for rows, cols in [(2, 4), (4, 8), (8, 16)]:
        W = np.zeros((rows, cols), dtype=np.float16)
        value = 1
        for i in range(rows):
            for j in range(0, cols, 4):
                W[i, j] = value
                W[i, j + 2] = value + 1
                value += 2
        cases.append(W)

    density_ok = 1.0
    ratio_ok = 1.0

    for W in cases:
        expected_density, expected_packed, expected_ratio = _oracle(W)
        try:
            got_density, got_packed = sol.measure_24_reduction(W)
            got_density = float(got_density)
            got_packed = int(got_packed)
        except Exception:
            density_ok = 0.0
            ratio_ok = 0.0
            break

        if got_density != expected_density:
            density_ok = 0.0

        if got_packed <= 0:
            ratio_ok = 0.0
            continue

        got_ratio = float(W.nbytes / got_packed)
        if abs(got_ratio - expected_ratio) > 1e-6 or got_packed != expected_packed:
            ratio_ok = 0.0

    return {
        "size_ratio": ratio_ok,
        "density": density_ok,
    }
