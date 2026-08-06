def blockwise_quantize(tensor, block_size, bits=8):
    raise NotImplementedError


def blockwise_dequantize(quantized, scales, block_size, original_len, bits=8):
    raise NotImplementedError
