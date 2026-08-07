import numpy as np
from bnb_sim.quant import quantize_int8, dequantize_int8

def mixed_precision_matmul(A, B, threshold=6.0):
    qB, scalesB, outB = quantize_int8(B, threshold=threshold)
    B_approx = dequantize_int8(qB, scalesB, outB)
    return A @ B_approx
