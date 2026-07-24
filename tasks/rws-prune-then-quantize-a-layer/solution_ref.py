import numpy as np

def prune_then_quantize(W: np.ndarray, group_size: int) -> np.ndarray:
    """
    Prune a 2‑D weight matrix with a 2:4 mask and then quantize the surviving
    entries per block using signed int‑4 scaling. The result is dequantized.
    """
    W = np.asarray(W, dtype=np.float64)
    n, d = W.shape
    out = np.zeros_like(W)

    for i in range(n):
        row = W[i]
        for start in range(0, d, group_size):
            end = min(start + group_size, d)
            block = row[start:end].copy()

            # keep only two largest‑abs entries per block
            if len(block) <= 2:
                keep_idx = np.arange(len(block))
            else:
                keep_idx = np.argpartition(np.abs(block), -2)[-2:]

            mask = np.zeros_like(block, dtype=bool)
            mask[keep_idx] = True
            block[~mask] = 0.0

            max_abs = np.max(np.abs(block))
            if max_abs == 0:
                out[i, start:end] = 0.0
                continue

            scale = max_abs / 7.0
            q = np.round(block / scale).clip(-8, 7)
            out[i, start:end] = q * scale

    return out
