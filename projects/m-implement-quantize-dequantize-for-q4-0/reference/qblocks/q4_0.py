import numpy as np


def quantize_q4_0(data: np.ndarray) -> list[dict]:
    data = np.asarray(data, dtype=np.float32).ravel()
    assert len(data) % 32 == 0
    blocks = []
    for i in range(0, len(data), 32):
        chunk = data[i:i + 32]
        amax = np.max(np.abs(chunk))
        d = float(amax / -8.0) if amax != 0 else 0.0
        qs = np.zeros(16, dtype=np.uint8)
        if d != 0.0:
            id_val = 1.0 / d
            for j in range(32):
                q = int(round(chunk[j] * id_val)) + 8
                q = max(0, min(15, q))
                byte_idx = j % 16
                if j < 16:
                    qs[byte_idx] |= (q & 0x0F)
                else:
                    qs[byte_idx] |= ((q & 0x0F) << 4)
        blocks.append({"d": float(np.float16(d)), "qs": qs})
    return blocks


def dequantize_q4_0(blocks: list[dict]) -> np.ndarray:
    out = np.zeros(len(blocks) * 32, dtype=np.float32)
    for idx, b in enumerate(blocks):
        d = float(np.float16(b["d"]))
        qs = b["qs"]
        for j in range(16):
            low = int(qs[j] & 0x0F) - 8
            high = int((qs[j] >> 4) & 0x0F) - 8
            out[idx * 32 + j] = low * d
            out[idx * 32 + j + 16] = high * d
    return out
