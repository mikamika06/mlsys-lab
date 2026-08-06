import numpy as np
from gguf_quant.scale import unpack_scales_and_mins

def dequantize_q4_k_block(block_bytes: bytes) -> np.ndarray:
    assert len(block_bytes) == 144
    d = np.frombuffer(block_bytes[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(block_bytes[2:4], dtype=np.float16)[0].astype(np.float32)
    scales, mins = unpack_scales_and_mins(block_bytes[4:16])
    qs = np.frombuffer(block_bytes[16:144], dtype=np.uint8)
    out = np.zeros(256, dtype=np.float32)
    for i in range(8):
        s = scales[i]
        m = mins[i]
        sub_qs = qs[i * 16 : (i + 1) * 16]
        low = sub_qs & 0x0F
        high = sub_qs >> 4
        chunk = np.empty(32, dtype=np.float32)
        chunk[0::2] = low
        chunk[1::2] = high
        out[i * 32 : (i + 1) * 32] = d * s * chunk - dmin * m
    return out
