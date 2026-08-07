import math

def tiled_causal_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], tile_q: int, tile_kv: int, on_tile=None) -> list[list[float]]:
    """Tiled causal attention that skips fully-future KV tiles entirely.

    See task.md for the exact skip rule and masking. Calls
    on_tile(qi, kj) exactly once per visited (non-skipped) tile pair.
    """
    raise NotImplementedError('your code here')
