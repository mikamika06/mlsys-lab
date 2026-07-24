import numpy as np


def tiled_vae_decode(z: np.ndarray, decode_fn, tile_size: int, overlap: int) -> np.ndarray:
    """
    Decode `z` (H, W, Cin) in overlapping tiles using `decode_fn` (which
    refuses tiles bigger than tile_size + 2*overlap), blending adjacent
    tiles' outputs with a linear ramp over the overlap band. Returns the
    reconstructed (H, W, Cout) image. See task.md.
    """
    raise NotImplementedError('your code here')
