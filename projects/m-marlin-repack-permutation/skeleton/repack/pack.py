import numpy as np


def pack_int4_standard(W):
    """Pack 2D uint8 int4 weights into standard GPTQ uint32 format along K dim."""
    raise NotImplementedError


def unpack_int4_standard(packed_W, K, N):
    """Unpack standard GPTQ uint32 matrix back into uint8 int4 weights."""
    raise NotImplementedError


def repack_gptq_to_marlin(packed_gptq, K, N):
    """Repack standard GPTQ packed uint32 matrix into Marlin packed format."""
    raise NotImplementedError
