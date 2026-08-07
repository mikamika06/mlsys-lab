import numpy as np
from quantizer.core import quantize_and_dequantize
from quantizer.qparams import derive_qparams


def test_subvector_boundary_handling():
    arr = np.arange(64, dtype=np.float32).reshape(4, 16)
    arr[0, :] *= 10.0
    arr[1, :] *= 0.1

    scale, zp = derive_qparams(arr, "sub_vector", symmetric=True, num_bits=8, group_size=4)
    assert scale.shape == (4, 4), f"Expected scale shape (4, 4), got {scale.shape}"

    deq = quantize_and_dequantize(arr, "sub_vector", symmetric=True, num_bits=8, group_size=4)
    assert deq.shape == arr.shape, f"Expected output shape {arr.shape}, got {deq.shape}"

    max_err = np.max(np.abs(arr - deq))
    assert max_err < 5.0, f"Max absolute error too high: {max_err}"
