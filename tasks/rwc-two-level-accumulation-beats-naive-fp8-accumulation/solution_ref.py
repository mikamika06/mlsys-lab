import numpy as np


def two_level_accumulate(x: np.ndarray, block_size: int) -> float:
    """
    Two-level accumulation over a long sequence.

    Level 1 (per block): accumulate elements sequentially, rounding the
    running total down to float16 after every addition -- this simulates a
    low-precision (FP8-like) accumulator register, but its error is bounded
    because each block only ever holds a short run.

    Level 2 (across blocks): promote each block's final float16 total to
    float32 ("dequantize") and fold it into a float32 grand total, which
    never loses precision to swamping.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    total = np.float32(0.0)
    for start in range(0, n, block_size):
        block = x[start:start + block_size]
        block_acc = np.float16(0.0)
        for v in block:
            block_acc = np.float16(np.float32(block_acc) + np.float32(v))
        total = np.float32(total) + np.float32(block_acc)
    return float(total)
