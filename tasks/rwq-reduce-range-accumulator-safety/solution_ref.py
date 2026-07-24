import numpy as np

def reduce_range_accumulator_safety(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per‑column accumulations for full and reduced activation ranges,
    and the maximum intermediate partial sum when using the reduced range.
    All outputs are int32 arrays of shape (C,).
    """
    # Full‑range accumulation
    full_accum = np.sum(X, axis=0, dtype=np.int32)

    # Reduced‑range: clamp to 63
    reduced_X = np.minimum(X, 63).astype(np.uint8)
    reduced_accum = np.sum(reduced_X, axis=0, dtype=np.int32)

    # Peak intermediate partial sum per column (reduced range)
    peak_per_col = np.max(
        np.cumsum(reduced_X.astype(np.int64), axis=0),
        axis=0
    ).astype(np.int32)

    return full_accum, reduced_accum, peak_per_col
