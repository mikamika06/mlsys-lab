import numpy as np

ROW_PERM = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]


def get_marlin_perm_map(K, N):
    """Return index permutation map for Marlin weight layout."""
    raise NotImplementedError


def permute_weights(W):
    """Permute uint8 weights into Marlin tile order."""
    raise NotImplementedError


def unpermute_weights(W_perm, K, N):
    """Restore original weight layout from Marlin permuted layout."""
    raise NotImplementedError
