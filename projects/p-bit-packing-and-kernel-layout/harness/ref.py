import numpy as np
from quant.packing import pack_bits, unpack_bits, simulate_kernel
from quant.layout import describe_layout, transform_layout

def get_oracle_packed(tensor, bits=4):
    return pack_bits(tensor, bits=bits)

def get_oracle_unpacked(packed, bits=4, shape=None):
    return unpack_bits(packed, bits=bits, shape=shape)

def get_oracle_kernel(packed, scale):
    return simulate_kernel(packed, scale)
