from enum import Enum
from typing import Optional
import numpy as np


class ErrorCategory(Enum):
    NONE = "none"
    RANK_MISMATCH = "rank_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    HEAD_DIM_UNSUPPORTED = "head_dim_unsupported"
    NON_CONTIGUOUS = "non_contiguous"
    MISALIGNED_POINTER = "misaligned_pointer"
    SHAPE_MISMATCH = "shape_mismatch"


class InputContractError(Exception):
    def __init__(self, category: ErrorCategory, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.category = category
        self.message = message
        self.details = details or {}


def _is_aligned(arr: np.ndarray, alignment_bytes: int) -> bool:
    ptr = arr.__array_interface__["data"][0]
    if ptr % alignment_bytes != 0:
        return False
    itemsize = arr.dtype.itemsize
    stride_bytes = [s * itemsize for s in arr.strides]
    for idx, s in enumerate(stride_bytes):
        if arr.shape[idx] > 1 and s % alignment_bytes != 0:
            return False
    return True


def classify_input_error(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alignment_bytes: int = 16,
    max_head_dim: int = 256,
) -> ErrorCategory:
    tensors = [("q", q), ("k", k), ("v", v)]
    for name, t in tensors:
        if t.ndim != 4:
            return ErrorCategory.RANK_MISMATCH

    q_dtype = q.dtype
    if k.dtype != q_dtype or v.dtype != q_dtype:
        return ErrorCategory.DTYPE_MISMATCH

    b, s_q, h_q, d = q.shape
    b_k, s_k, h_k, d_k = k.shape
    b_v, s_v, h_v, d_v = v.shape

    if b != b_k or b != b_v:
        return ErrorCategory.SHAPE_MISMATCH
    if h_q != h_k or h_q != h_v:
        return ErrorCategory.SHAPE_MISMATCH
    if d != d_k or d != d_v:
        return ErrorCategory.SHAPE_MISMATCH
    if s_k != s_v:
        return ErrorCategory.SHAPE_MISMATCH

    if d % 8 != 0 or d > max_head_dim or d == 0:
        return ErrorCategory.HEAD_DIM_UNSUPPORTED

    for name, t in tensors:
        if not t.flags["C_CONTIGUOUS"]:
            return ErrorCategory.NON_CONTIGUOUS

    for name, t in tensors:
        if not _is_aligned(t, alignment_bytes):
            return ErrorCategory.MISALIGNED_POINTER

    return ErrorCategory.NONE
