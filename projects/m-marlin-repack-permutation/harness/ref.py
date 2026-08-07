import numpy as np

ROW_PERM = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]

CONFIGS = [
    (16, 64),
    (32, 64),
    (32, 128),
    (64, 256),
]


def get_marlin_perm_map(K, N):
    if K % 16 != 0 or N % 64 != 0:
        raise ValueError("K must be multiple of 16 and N multiple of 64")
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
    W = np.asarray(W)
    K, N = W.shape
    perm = get_marlin_perm_map(K, N)
    return W.reshape(-1)[perm].reshape(K, N)


def unpermute_weights(W_perm, K, N):
    perm = get_marlin_perm_map(K, N)
    inv_perm = np.argsort(perm)
    return np.asarray(W_perm).reshape(-1)[inv_perm].reshape(K, N)


def pack_int4_standard(W):
    W = np.asarray(W, dtype=np.uint8)
    K, N = W.shape
    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for p in range(8):
        packed |= (W[p::8, :].astype(np.uint32) & 0xF) << (4 * p)
    return packed


def unpack_int4_standard(packed_W, K, N):
    packed_W = np.asarray(packed_W, dtype=np.uint32)
    W = np.zeros((K, N), dtype=np.uint8)
    for p in range(8):
        W[p::8, :] = ((packed_W >> (4 * p)) & 0xF).astype(np.uint8)
    return W


def repack_gptq_to_marlin(packed_gptq, K, N):
    W = unpack_int4_standard(packed_gptq, K, N)
    W_perm = permute_weights(W)
    flat = W_perm.reshape(-1)
    n_uint32 = flat.size // 8
    marlin = np.zeros(n_uint32, dtype=np.uint32)
    for p in range(8):
        marlin |= (flat[p::8].astype(np.uint32) & 0xF) << (4 * p)
    return marlin.reshape(K // 16, N * 2)


def marlin_to_gptq(marlin_packed, K, N):
    marlin_packed = np.asarray(marlin_packed, dtype=np.uint32)
    flat_packed = marlin_packed.reshape(-1)
    flat_w = np.zeros(K * N, dtype=np.uint8)
    for p in range(8):
        flat_w[p::8] = ((flat_packed >> (4 * p)) & 0xF).astype(np.uint8)
    W_perm = flat_w.reshape(K, N)
    W = unpermute_weights(W_perm, K, N)
    return pack_int4_standard(W)


def generate_test_matrix(K, N, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 16, size=(K, N), dtype=np.uint8)
