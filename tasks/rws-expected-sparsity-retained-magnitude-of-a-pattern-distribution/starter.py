import numpy as np


def expected_pattern_stats(p: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group expected density and expected retained sum |w| under a
    probability distribution over the 6 canonical 2-of-4 keep patterns.

    p: (G, 6) probability rows (sum to 1) over the patterns.
    w: (G, 4) absolute weight magnitudes.
    Returns (expected_density, expected_retained), each shape (G,).
    """
    raise NotImplementedError('your code here')
