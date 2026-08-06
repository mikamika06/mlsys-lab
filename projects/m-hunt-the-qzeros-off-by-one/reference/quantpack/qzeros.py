import numpy as np

def decode_qzeros(packed_z: np.ndarray, num_groups: int, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    out = np.zeros(num_groups, dtype=np.float32)
    shift_per_byte = 8 // bits
    for i in range(num_groups):
        byte_idx = i // shift_per_byte
        bit_offset = (i % shift_per_byte) * bits
        val = (packed_z[byte_idx] >> bit_offset) & mask
        out[i] = float(val + 1)
    return out
