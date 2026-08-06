import math

def layer_bytes(shape, method):
    """
    Return the byte size of a layer for a given method.
    Methods:
    - float16
    - int8_channel
    - lut4_channel_fp16
    - lut4_joint_int8_channel
    - lut8_channel_fp16
    - lut8_joint_int8_channel
    """
    raise NotImplementedError

def optimize_model(shapes, allowed_methods):
    """
    Return (plan, total_bytes) where plan is a list of method names (one per shape).
    In case of ties, prefer the first method in allowed_methods.
    """
    raise NotImplementedError
