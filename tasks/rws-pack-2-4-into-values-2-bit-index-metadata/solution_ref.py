import numpy as np

def pack_2_of_4(a: np.ndarray):
    n = a.shape[0]
    if n % 4 != 0:
        raise ValueError("Length must be divisible by 4")
    blocks = a.reshape(-1, 4)
    abs_blocks = np.abs(blocks)
    # indices of two largest magnitudes per block
    top2_idx = np.argsort(abs_blocks, axis=1)[:, ::-1][:, :2]
    # sort within each block to preserve left‑to‑right order
    sorted_top2 = np.sort(top2_idx, axis=1)
    values = blocks[np.arange(blocks.shape[0])[:, None], sorted_top2].ravel()
    indices = sorted_top2.ravel().astype(np.uint8)
    return values.astype(np.float64), indices
