import numpy as np


def nvfp4_reconstruct(global_scale, e4m3_block_codes: np.ndarray, e2m1_codes: np.ndarray) -> np.ndarray:
    """Decode e4m3_block_codes (n_blocks,) uint8 -> per-block scale s_b
    (S:1,E:4,M:3, bias 7). Decode e2m1_codes (n_blocks,16) uint8 -> per-
    element value q (S:1,E:2,M:1, bias 1). Return global_scale * s_b[:,None]
    * q, shape (n_blocks, 16)."""
    raise NotImplementedError('your code here')
