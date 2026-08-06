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


def classify_input_error(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alignment_bytes: int = 16,
    max_head_dim: int = 256,
) -> ErrorCategory:
    raise NotImplementedError
