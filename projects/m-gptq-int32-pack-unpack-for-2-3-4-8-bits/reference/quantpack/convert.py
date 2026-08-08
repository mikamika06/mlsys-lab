import numpy as np

def convert_awq_to_gptq(awq_packed, bits, original_shape):
    from quantpack.packing import unpack_weights, pack_weights
    unpacked = unpack_weights(awq_packed, bits, original_shape)
    transposed = unpacked.T
    return pack_weights(transposed, bits)
