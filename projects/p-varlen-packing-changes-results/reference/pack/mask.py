import numpy as np


def make_causal_mask(cu_seqlens, max_len):
    mask = np.full((max_len, max_len), -10000.0, dtype=np.float32)
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        for r in range(s, e):
            for c in range(s, e):
                if c <= r:
                    mask[r, c] = 0.0
    return mask


def check_equivalence(packed_out, unpacked_list):
    idx = 0
    for u in unpacked_list:
        length = len(u)
        segment = packed_out[idx:idx + length]
        if not np.allclose(segment, u, atol=1e-5):
            return False
        idx += length
    return True
