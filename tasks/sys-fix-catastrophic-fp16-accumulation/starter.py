import numpy as np


def sum_rows_fp32(A: np.ndarray) -> np.ndarray:
    # TODO: fp16 accumulation destroys information during long reductions.
    return np.sum(A, axis=1, dtype=np.float16).astype(np.float32)
