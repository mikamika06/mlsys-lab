import numpy as np


def dequantize_q2_k(block_bytes: bytes) -> np.ndarray:
    """Dequantizes a single Q2_K block (256 weights) into float32 array."""
    dt = np.dtype([
        ('scales', np.uint8, 16),
        ('qs', np.uint8, 64),
        ('d', np.float16),
        ('dmin', np.float16)
    ])
    block = np.frombuffer(block_bytes, dtype=dt)[0]

    scales = block['scales']
    qs = block['qs']
    d = float(block['d'])
    dmin = float(block['dmin'])

    weights = np.zeros(256, dtype=np.float32)

    for i in range(16):
        sc = scales[i]
        d_sub = d * (sc & 0x0F)
        m_sub = dmin * (sc >> 4)

        shift = (i // 4) * 2
        byte_offset = (i % 4) * 16

        for j in range(16):
            q_byte = qs[byte_offset + j]
            q_val = (q_byte >> shift) & 0x03
            weights[i * 16 + j] = d_sub * q_val - m_sub

    return weights


def reconstruct_q3_k_scales(hmask: bytes, scales_raw: bytes) -> np.ndarray:
    """Reconstructs 16 6-bit scales for a Q3_K block using raw scales and hmask."""
    hm = np.frombuffer(hmask, dtype=np.uint8)
    sc = np.frombuffer(scales_raw, dtype=np.uint8)

    scales = np.zeros(16, dtype=np.int8)

    for i in range(16):
        low4 = sc[i] & 0x0F
        high2 = (hm[i // 2] >> ((i % 2) * 4)) & 0x03
        scales[i] = int(low4 | (high2 << 4)) - 32

    return scales
