import numpy as np


def tiled_causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                            tile_q: int, tile_kv: int, on_tile=None) -> np.ndarray:
    """Tiled causal attention that skips fully-future KV tiles entirely.

    See task.md for the exact skip rule and masking. Calls
    on_tile(qi, kj) exactly once per visited (non-skipped) tile pair.
    """
    raise NotImplementedError('your code here')
