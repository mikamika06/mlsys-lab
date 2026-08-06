import numpy as np

def max_abs_error(direct, onnx):
    return float(np.max(np.abs(direct - onnx)))
