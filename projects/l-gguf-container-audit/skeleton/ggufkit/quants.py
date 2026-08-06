import struct

QK_K = 256


def half_to_float(bits):
    """An IEEE-754 binary16 given as a 16-bit integer, as a Python float.

    Subnormals and zero included. struct is allowed for reading bytes; the
    conversion itself is yours.
    """
    raise NotImplementedError


def dequant_q4_k(block):
    """One 144-byte Q4_K superblock as 256 floats.

    Layout: d (half), dmin (half), 12 bytes of packed 6-bit scales and mins,
    then 128 bytes holding two 4-bit quants each. The block is eight groups of
    32; group j takes its scale and min from the packed 12 bytes.
    """
    raise NotImplementedError


def dequant_q6_k(block):
    """One 210-byte Q6_K superblock as 256 floats.

    Layout: 128 bytes of low nibbles, 64 bytes holding the high 2 bits, 16
    signed 8-bit scales, then d (half). Each quant is centred by subtracting 32.
    """
    raise NotImplementedError


def dequant_f32(block):
    raise NotImplementedError


def dequant_f16(block):
    raise NotImplementedError


def dequant_tensor(raw, type_id, n_elements):
    """A whole tensor, row-major, as a flat list of floats."""
    raise NotImplementedError
