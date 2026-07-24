import numpy as np


def q4_0_block_pack_unpack(x: np.ndarray) -> dict:
    """ggml Q4_0: x is 1-D, length a multiple of 32. Per 32-block, let
    x_star be the signed element of largest |.|, d = x_star / -8, cast to
    float16 (d16). nibble_i = clip(round(x_i/d16)+8, 0, 15). dequant_i =
    (nibble_i - 8) * d16. Return {"scale": (n_blocks,) float16,
    "nibbles": (n_blocks,32) uint8, "dequant": (n_blocks,32) float64}."""
    raise NotImplementedError('your code here')
