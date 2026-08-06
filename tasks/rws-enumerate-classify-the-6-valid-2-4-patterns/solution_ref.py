import numpy as np

def _build_reference():
    patterns = [tuple(int(b) for b in format(i, '04b')) for i in range(16)]
    valid = sorted([p for p in patterns if sum(p) == 2])
    return {p: idx for idx, p in enumerate(valid)}

_REFERENCE_MAP = _build_reference()

def classify_patterns(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=int)
    n_rows = vectors.shape[0]
    result = np.full(n_rows, -1, dtype=int)

    for i in range(n_rows):
        row = vectors[i]
        row_sum = 0
        for val in row:
            row_sum += val
        if row_sum == 2:
            t = tuple(row)
            if t in _REFERENCE_MAP:
                result[i] = _REFERENCE_MAP[t]

    return result
