import numpy as np


def verify_quantization_operation(tensor, quantized, scale):
    arr = np.array(tensor, dtype=np.float32)
    q_arr = np.array(quantized, dtype=np.float32)
    err_direct = np.mean(np.abs(arr * scale - q_arr))
    err_inverse = np.mean(np.abs(arr / scale - q_arr))
    if err_inverse < err_direct:
        return "inverted"
    return "correct"
