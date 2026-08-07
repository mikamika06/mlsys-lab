import numpy as np
from reference.quant.codebook import get_nf4_codebook, get_fp4_codebook
from reference.quant.quantize import quantize_blockwise, dequantize_blockwise

def compute_reference_codebooks():
    return get_nf4_codebook(), get_fp4_codebook()

def compute_reference_quant(weights, block_size, fmt):
    return quantize_blockwise(weights, block_size, fmt)

def compute_reference_dequant(q, scales, block_size, fmt, shape):
    return dequantize_blockwise(q, scales, block_size, fmt, shape)
