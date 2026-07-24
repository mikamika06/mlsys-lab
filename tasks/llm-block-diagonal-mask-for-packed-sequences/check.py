import numpy as np
from mlsys.scorers import byte_exact_fraction

def _ref(seq_lengths):
    total = sum(seq_lengths)
    mask = np.zeros((total, total), dtype=bool)
    start = 0
    for l in seq_lengths:
        end = start + l
        mask[start:end, start:end] = True
        start = end
    return mask

def grade(sol, fx) -> dict:
    cases = [
        [3, 2],
        [5],
        [1, 4, 2],
        [2, 2, 2],
        [],
    ]
    ok = 1.0
    for seq_lengths in cases:
        try:
            got = sol.packed_block_diagonal_mask(seq_lengths)
            ref = _ref(seq_lengths)
        except Exception:
            return {"byte_exact_fraction": 0.0}
        if not isinstance(got, np.ndarray):
            return {"byte_exact_fraction": 0.0}
        if got.shape != ref.shape or got.dtype != ref.dtype:
            return {"byte_exact_fraction": 0.0}
        # Handle empty mask specially: byte_exact_fraction returns 0 for empty arrays
        if ref.size == 0:
            frac = 1.0
        else:
            frac = byte_exact_fraction(ref.tobytes(), got.tobytes())
        if frac < 1.0 - 1e-12:
            ok = 0.0
            break
    return {"byte_exact_fraction": ok}
