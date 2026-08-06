import numpy as np


def gather_kv_cache(
    physical_blocks: np.ndarray,
    block_table: list[int],
    seq_len: int,
) -> np.ndarray:
    """Gathers contiguous logical KV cache tensor from physical block storage."""
    if seq_len == 0 or not block_table:
        shape = (0,) + physical_blocks.shape[2:]
        return np.empty(shape, dtype=physical_blocks.dtype)

    block_size = physical_blocks.shape[1]
    gathered_blocks = physical_blocks[block_table]
    logical_tensor = gathered_blocks.reshape((-1,) + physical_blocks.shape[2:])
    return logical_tensor[:seq_len]
