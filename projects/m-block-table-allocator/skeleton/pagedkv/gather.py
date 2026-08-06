import numpy as np


def gather_kv_cache(
    physical_blocks: np.ndarray,
    block_table: list[int],
    seq_len: int,
) -> np.ndarray:
    """Gathers contiguous logical KV cache tensor from physical block storage."""
    raise NotImplementedError
