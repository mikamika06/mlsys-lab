import numpy as np
from quant.core import get_view, restore_view, calc_qparams, apply_quant, apply_dequant

def evaluate_ladder(w, group_size=32):
    """
    Run asymmetric quantization across all four granularities
    ("tensor", "axis_0", "axis_1", "group").
    For each, calculate max_abs_err against the original `w` and metadata size.
    Assume scales cost 2 bytes each, zero-points cost 1 byte each.
    Return a list of dictionaries with:
    - granularity (str)
    - meta_bytes (int)
    - max_abs_err (float)
    """
    raise NotImplementedError
