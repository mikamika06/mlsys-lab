import numpy as np
from typing import Tuple

def block_stats(seqlens: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    num_blocks = (seqlens + block_size - 1) // block_size
    slack = num_blocks * block_size - seqlens
    return num_blocks.astype(np.int64), slack.astype(np.int64)
