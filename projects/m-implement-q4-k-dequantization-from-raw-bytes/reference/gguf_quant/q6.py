import numpy as np

def dequantize_q6_k_block(block_bytes: bytes) -> np.ndarray:
    assert len(block_bytes) == 210
    ql = np.frombuffer(block_bytes[0:128], dtype=np.uint8)
    qh = np.frombuffer(block_bytes[128:192], dtype=np.uint8)
    scales = np.frombuffer(block_bytes[192:208], dtype=np.int8).astype(np.float32)
    d = np.frombuffer(block_bytes[208:210], dtype=np.float16)[0].astype(np.float32)
    out = np.zeros(256, dtype=np.float32)
    for i in range(16):
        sc = scales[i]
        for j in range(16):
            idx = i * 16 + j
            byte_idx_l = idx // 2
            if idx % 2 == 0:
                l_val = ql[byte_idx_l] & 0x0F
            else:
                l_val = ql[byte_idx_l] >> 4
            byte_idx_h = idx // 4
            shift = (idx % 4) * 2
            h_val = (qh[byte_idx_h] >> shift) & 0x03
            q = l_val | (h_val << 4)
            out[idx] = d * sc * (q - 32.0)
    return out
