import itertools

import numpy as np


def _pattern_masks() -> np.ndarray:
    """The 6 canonical 2-of-4 keep patterns, in lexicographic (i,j) order."""
    masks = np.zeros((6, 4), dtype=np.float64)
    for k, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        masks[k, i] = 1.0
        masks[k, j] = 1.0
    return masks


_MASKS = _pattern_masks()


def expected_pattern_stats(p: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group expected density and expected retained sum |w| under a
    probability distribution over the 6 canonical 2-of-4 keep patterns.

    p: (G, 6) probability rows (sum to 1) over the patterns.
    w: (G, 4) absolute weight magnitudes.
    Returns (expected_density, expected_retained), each shape (G,).
    """
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    density = p @ (_MASKS.sum(axis=1) / 4.0)         # (G,)
    marginal_keep = p @ _MASKS                        # (G,4) marginal keep-prob per position
    retained = np.sum(marginal_keep * w, axis=1)       # (G,)

    return density, retained
