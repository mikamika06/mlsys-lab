import numpy as np


def _oracle_size(weight, group_size):
    flat = np.asarray(weight).reshape(-1)
    groups = flat.size // group_size
    total = 0.0
    for g in range(groups):
        block = flat[g * group_size:(g + 1) * group_size]
        nnz = int(np.count_nonzero(block))
        total += nnz * (4 / 8)
        total += nnz * (2 / 8)
        total += 2.0
    return float(total)


def grade(sol, fx) -> dict:
    cases = []
    rng = np.random.default_rng(7)

    for rows, cols in [(4, 8), (2, 12), (8, 16)]:
        size = rows * cols
        arr = np.zeros(size, dtype=np.float32)
        for start in range(0, size, 4):
            chosen = rng.choice(4, size=2, replace=False)
            arr[start + chosen[0]] = rng.normal()
            arr[start + chosen[1]] = rng.normal()
        cases.append((arr.reshape(rows, cols), 4))

    ok = 1.0
    for weight, group_size in cases:
        ref_bytes = _oracle_size(weight, group_size)
        dense_bytes = float(np.asarray(weight).size * 2)
        ref_ratio = dense_bytes / ref_bytes
        try:
            got_bytes = float(sol.combined_served_size(weight, group_size))
            got_ratio = dense_bytes / got_bytes
        except Exception:
            ok = 0.0
            break
        if abs(got_ratio - ref_ratio) > 1e-6:
            ok = 0.0
            break

    return {"size_ratio": ok}
