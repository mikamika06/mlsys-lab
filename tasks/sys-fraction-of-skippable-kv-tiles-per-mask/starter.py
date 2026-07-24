import numpy as np


def skippable_kv_tile_fraction(mask: np.ndarray, tile_size: int) -> float:
    """Fraction of (query-tile, KV-tile) blocks in `mask` that are entirely
    False -- i.e. blocks a tiled/blocked attention kernel could skip
    computing entirely because no (query, key) pair inside them is
    attended to.

    `mask` is an (n, n) boolean array, True meaning "attention allowed".
    `n` is divisible by `tile_size`.
    """
    raise NotImplementedError('your code here')
