import numpy as np
from quant.symmetric import quantize_symmetric, dequantize_symmetric


def quantize_per_tensor(weights, qmin=-128, qmax=127):
    q, scale = quantize_symmetric(weights, qmin, qmax)
    deq = dequantize_symmetric(q, scale)
    return q, scale, deq


def quantize_per_channel(weights, qmin=-128, qmax=127, axis=0):
    slices = [slice(None)] * weights.ndim
    scales = []
    q_list = []
    deq_list = []
    for i in range(weights.shape[axis]):
        slices[axis] = i
        sub = weights[tuple(slices)]
        q_sub, scale_sub = quantize_symmetric(sub, qmin, qmax)
        deq_sub = dequantize_symmetric(q_sub, scale_sub)
        scales.append(scale_sub)
        q_list.append(q_sub)
        deq_list.append(deq_sub)
    q = np.stack(q_list, axis=axis)
    deq = np.stack(deq_list, axis=axis)
    return q, np.array(scales, dtype=np.float32), deq


def measure_quantization_error(weights, dequantized_weights):
    return float(np.max(np.abs(weights - dequantized_weights)))
