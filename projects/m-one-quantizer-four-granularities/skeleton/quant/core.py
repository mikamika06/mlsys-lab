import numpy as np

def get_view(w, granularity, group_size=None):
    """
    Reshape a 2D weight matrix `w` (M, N) into a 2D view (Groups, Elements)
    based on the granularity: 'tensor', 'axis_0', 'axis_1', or 'group'.
    """
    raise NotImplementedError

def restore_view(q_view, w_shape, granularity):
    """
    Restore a 2D quantized view (Groups, Elements) back to the original
    weight matrix shape (M, N).
    """
    raise NotImplementedError

def calc_qparams(w_view, symmetric):
    """
    Calculate float scales and zero_points for a 2D weight view.
    If symmetric: int8 range [-127, 127], zp is 0.
    If asymmetric: uint8 range [0, 255], zp anchors 0.0 perfectly.
    Returns (scale, zp) both of shape (Groups, 1).
    """
    raise NotImplementedError

def apply_quant(w_view, scale, zp, symmetric):
    """
    Apply linear quantization to the view.
    Returns an int8 array if symmetric, or uint8 if asymmetric.
    """
    raise NotImplementedError

def apply_dequant(q_view, scale, zp):
    """
    Dequantize the integer view back to float32.
    """
    raise NotImplementedError
