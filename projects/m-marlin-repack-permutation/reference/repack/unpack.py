import numpy as np
from repack.pack import pack_int4_standard
from repack.permute import unpermute_weights


def marlin_to_gptq(marlin_packed, K, N):
    """Convert Marlin packed uint32 tensor back to standard GPTQ uint32 packed tensor."""
    marlin_packed = np.asarray(marlin_packed, dtype=np.uint32)
    flat_packed = marlin_packed.reshape(-1)
    flat_w = np.zeros(K * N, dtype=np.uint8)
    for p in range(8):
        flat_w[p::8] = ((flat_packed >> (4 * p)) & 0xF).astype(np.uint8)
    W_perm = flat_w.reshape(K, N)
    W = unpermute_weights(W_perm, K, N)
    return pack_int4_standard(W)
