import numpy as np


def quantize_q5_1(data: np.ndarray) -> list[dict]:
    data = np.asarray(data, dtype=np.float32).ravel()
    assert len(data) % 32 == 0
    blocks = []
    for i in range(0, len(data), 32):
        chunk = data[i:i + 32]
        min_val = float(np.min(chunk))
        max_val = float(np.max(chunk))
        d = (max_val - min_val) / 31.0 if max_val != min_val else 0.0
        m = min_val
        qs = np.zeros(16, dtype=np.uint8)
        qh_bytes = np.zeros(4, dtype=np.uint8)
        if d != 0.0:
            id_val = 1.0 / d
            for j in range(32):
                q = int(round((chunk[j] - m) * id_val))
                q = max(0, min(31, q))
                byte_idx = j % 16
                if j < 16:
                    qs[byte_idx] |= (q & 0x0F)
                else:
                    qs[byte_idx] |= ((q & 0x0F) << 4)
                if q & 0x10:
                    qh_bytes[j // 8] |= (1 << (j % 8))
        qh = int(qh_bytes[0]) | (int(qh_bytes[1]) << 8) | (int(qh_bytes[2]) << 16) | (int(qh_bytes[3]) << 24)
        blocks.append({"d": float(np.float16(d)), "m": float(np.float16(m)), "qh": qh, "qs": qs})
    return blocks


def dequantize_q5_1(blocks: list[dict]) -> np.ndarray:
    out = np.zeros(len(blocks) * 32, dtype=np.float32)
    for idx, b in enumerate(blocks):
        d = float(np.float16(b["d"]))
        m = float(np.float16(b["m"]))
        qh = int(b["qh"])
        qs = b["qs"]
        for j in range(16):
            low_qh = (qh >> j) & 1
            high_qh = (qh >> (j + 16)) & 1
            low = (int(qs[j] & 0x0F)) | (low_qh << 4)
            high = (int((qs[j] >> 4) & 0x0F)) | (high_qh << 4)
            out[idx * 32 + j] = low * d + m
            out[idx * 32 + j + 16] = high * d + m
    return out
