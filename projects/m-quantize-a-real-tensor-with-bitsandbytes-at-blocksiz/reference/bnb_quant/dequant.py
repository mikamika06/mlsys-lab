import numpy as np


def dequantize_blockwise(quant_dict, block_size):
    quantized = quant_dict["quantized"].astype(np.float32)
    absmax = quant_dict["absmax"][:, np.newaxis]
    dequant_blocks = (quantized / 127.0) * absmax
    flattened = dequant_blocks.flatten()
    orig_shape = quant_dict["original_shape"]
    total_elements = int(np.prod(orig_shape))
    result = flattened[:total_elements].reshape(orig_shape)
    return result
