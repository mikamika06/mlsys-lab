import numpy as np
from repack.permute import permute_weights


def pack_int4_standard(W):
    """Pack 2D uint8 int4 weights into standard GPTQ uint32 format along K dim."""
    W = np.asarray(W, dtype=np.uint8)
    K, N = W.shape
    packed = np.zeros((K // 8, N), dtype=np.uint32)
    for p in range(8):
        packed |= (W[p::8, :].astype(np.uint32) & 0xF) << (4 * p)
    return packed


def unpack_int4_standard(packed_W, K, N):
    """Unpack standard GPTQ uint32 matrix back into uint8 int4 weights."""
    packed_W = np.asarray(packed_W, dtype=np.uint32)
    W = np.zeros((K, N), dtype=np.uint8)
    for p in range(8):
        W[p::8, :] = ((packed_W >> (4 * p)) & 0xF).astype(np.uint8)
    return W


def repack_gptq_to_marlin(packed_gptq, K, N):
    """Repack standard GPTQ packed uint32 matrix into Marlin packed format."""
    W = unpack_int4_standard(packed_gptq, K, N)
    W_perm = permute_weights(W)
    flat = W_perm.reshape(-1)
    n_uint32 = flat.size // 8
    marlin = np.zeros(n_uint32, dtype=np.uint32)
    for p in range(8):
        marlin |= (flat[p::8].astype(np.uint32) & 0xF) << (4 * p)
    return marlin.reshape(K // 16, N * 2)
