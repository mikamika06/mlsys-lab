import numpy as np


def q4_0_block_pack_unpack(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    blocks = x.reshape(-1, 32)
    B = blocks.shape[0]

    idx = np.argmax(np.abs(blocks), axis=1)
    max_signed = blocks[np.arange(B), idx]
    d = max_signed / -8.0
    d16 = d.astype(np.float16)
    dq = d16.astype(np.float64)

    safe_dq = np.where(dq == 0.0, 1.0, dq)
    q = np.round(blocks / safe_dq[:, None]) + 8.0
    nibbles = np.clip(q, 0, 15).astype(np.uint8)
    nibbles = np.where(dq[:, None] == 0.0, np.uint8(8), nibbles)

    dequant = (nibbles.astype(np.float64) - 8.0) * dq[:, None]

    return {"scale": d16, "nibbles": nibbles, "dequant": dequant}
