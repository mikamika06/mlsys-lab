import numpy as np

# Build the canonical mapping once at import time.
def _build_reference():
    patterns = [tuple(int(b) for b in format(i, '04b')) for i in range(16)]
    valid = sorted([p for p in patterns if sum(p) == 2])
    return {p: idx for idx, p in enumerate(valid)}

_REFERENCE_MAP = _build_reference()

def classify_patterns(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=int)
    n_rows = vectors.shape[0]
    result = np.full(n_rows, -1, dtype=int)

    mask = vectors.sum(axis=1) == 2
    if not mask.any():
        return result

    valid_vectors = vectors[mask]
    # Convert each valid row to a tuple and look up its index.
    indices = np.array([_REFERENCE_MAP[tuple(v)] for v in valid_vectors], dtype=int)
    result[np.where(mask)[0]] = indices
    return result
