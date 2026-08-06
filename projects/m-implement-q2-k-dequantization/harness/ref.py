import numpy as np


def generate_q2_k_block(seed=42):
    rng = np.random.default_rng(seed)
    scales = rng.integers(0, 256, size=16, dtype=np.uint8)
    qs = rng.integers(0, 256, size=64, dtype=np.uint8)
    d = np.float16(rng.uniform(0.1, 2.0))
    dmin = np.float16(rng.uniform(0.1, 2.0))
    raw = scales.tobytes() + qs.tobytes() + d.tobytes() + dmin.tobytes()
    return raw, scales, qs, float(d), float(dmin)


def dequantize_q2_k_ref(block_bytes: bytes) -> np.ndarray:
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


def reconstruct_q3_k_scales_ref(hmask: bytes, scales_raw: bytes) -> np.ndarray:
    hm = np.frombuffer(hmask, dtype=np.uint8)
    sc = np.frombuffer(scales_raw, dtype=np.uint8)

    scales = np.zeros(16, dtype=np.int8)
    for i in range(16):
        low4 = sc[i] & 0x0F
        high2 = (hm[i // 2] >> ((i % 2) * 4)) & 0x03
        scales[i] = int(low4 | (high2 << 4)) - 32

    return scales


def calculate_kquant_bpw_ref(quant_type: str) -> float:
    sizes = {
        "Q2_K": 2.5,
        "Q3_K": 3.4375,
        "Q4_K": 4.5625,
        "Q5_K": 5.6875,
        "Q6_K": 6.8203125
    }
    return sizes[quant_type.upper()]
