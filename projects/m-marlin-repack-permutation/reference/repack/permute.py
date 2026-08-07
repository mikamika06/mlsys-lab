import numpy as np

ROW_PERM = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]


def get_marlin_perm_map(K, N):
    """Return index permutation map for Marlin weight layout."""
    if K % 16 != 0 or N % 64 != 0:
        raise ValueError("K must be divisible by 16 and N by 64")
    n_tiles_k = K // 16
    n_tiles_n = N // 64
    perm = []
    for tk in range(n_tiles_k):
        for tn in range(n_tiles_n):
            for g in range(4):
                for r in range(16):
                    src_r = tk * 16 + ROW_PERM[r]
                    for c in range(16):
                        src_c = tn * 64 + g * 16 + c
                        perm.append(src_r * N + src_c)
    return np.array(perm, dtype=np.int64)


def permute_weights(W):
    """Permute uint8 weights into Marlin tile order."""
    W = np.asarray(W)
    K, N = W.shape
    perm = get_marlin_perm_map(K, N)
    return W.reshape(-1)[perm].reshape(K, N)


def unpermute_weights(W_perm, K, N):
    """Restore original weight layout from Marlin permuted layout."""
    perm = get_marlin_perm_map(K, N)
    inv_perm = np.argsort(perm)
    return np.asarray(W_perm).reshape(-1)[inv_perm].reshape(K, N)
