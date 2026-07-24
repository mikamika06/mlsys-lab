import numpy as np


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    """Split (T, D) into (num_heads, T, D // num_heads), head axis first.

    Required mapping:  out[h, t, j] == x[t, h * (D // num_heads) + j]

    Use explicit index arithmetic (nested loops over h, t, j) and plain element
    assignment. Do NOT use reshape / transpose / swapaxes / moveaxis -- those run
    in C and will fail the op_count gate.
    """
    raise NotImplementedError("your code here")


def merge_heads(heads: np.ndarray) -> np.ndarray:
    """Inverse of split_heads: (num_heads, T, head_dim) -> (T, num_heads * head_dim).

    Required mapping:  out[t, h * head_dim + j] == heads[h, t, j]

    Use explicit index arithmetic only -- no reshape / transpose / swapaxes.
    """
    raise NotImplementedError("your code here")
