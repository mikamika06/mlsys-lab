import numpy as np


def get_format_properties(dtype_str: str) -> dict:
    """Return numeric properties of floating point format.

    Supported dtype_str: 'fp16', 'bf16', 'fp32'
    Returns dict with keys:
        - exponent_bits (int)
        - mantissa_bits (int)
        - max_val (float)
        - min_pos_normal (float)
        - min_pos_subnormal (float)
        - eps (float)
    """
    raise NotImplementedError


def compute_relative_error(val: float, dtype_str: str) -> float:
    """Compute relative error |val - cast(val)| / |val| when casting to target format."""
    raise NotImplementedError
