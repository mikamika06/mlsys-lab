import numpy as np


def verify_numeric_agreement(direct_outputs, onnx_outputs, rtol=1e-5, atol=1e-5):
    if len(direct_outputs) != len(onnx_outputs):
        return False
    for d, o in zip(direct_outputs, onnx_outputs):
        d_arr = np.asarray(d, dtype=np.float32)
        o_arr = np.asarray(o, dtype=np.float32)
        if not np.allclose(d_arr, o_arr, rtol=rtol, atol=atol):
            return False
    return True
