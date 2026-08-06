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

    G = p.shape[0]

    mask_sums = np.zeros(6, dtype=np.float64)
    for k in range(6):
        row_sum = 0.0
        for c in range(4):
            row_sum += _MASKS[k, c]
        mask_sums[k] = row_sum / 4.0

    density = np.zeros(G, dtype=np.float64)
    for g in range(G):
        acc = 0.0
        for k in range(6):
            acc += p[g, k] * mask_sums[k]
        density[g] = acc

    marginal_keep = np.zeros((G, 4), dtype=np.float64)
    for g in range(G):
        for c in range(4):
            acc = 0.0
            for k in range(6):
                acc += p[g, k] * _MASKS[k, c]
            marginal_keep[g, c] = acc

    retained = np.zeros(G, dtype=np.float64)
    for g in range(G):
        acc = 0.0
        for c in range(4):
            acc += marginal_keep[g, c] * w[g, c]
        retained[g] = acc

    return density, retained
