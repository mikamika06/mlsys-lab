import numpy as np

def warp_divergence_branch_count(preds: np.ndarray, warp_size: int = 32) -> np.ndarray:
    preds = np.asarray(preds)
    if preds.ndim != 1:
        raise ValueError("preds must be a one‑dimensional array")
    n = len(preds)
    if n % warp_size != 0:
        raise ValueError(f"Length {n} is not a multiple of warp_size {warp_size}")
    num_blocks = n // warp_size
    out = np.empty(num_blocks, dtype=int)
    for i in range(num_blocks):
        block = preds[i * warp_size : (i + 1) * warp_size]
        unique_count = 0
        seen = []
        for item in block:
            is_new = True
            for s in seen:
                if s == item:
                    is_new = False
                    break
            if is_new:
                seen.append(item)
                unique_count += 1
        out[i] = unique_count
    return out
