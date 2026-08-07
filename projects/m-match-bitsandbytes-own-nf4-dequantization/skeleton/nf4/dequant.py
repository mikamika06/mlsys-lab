import numpy as np


def get_nf4_table():
    raise NotImplementedError


def unpack_4bit(packed_bytes):
    raise NotImplementedError


def dequantize_nf4(packed_bytes, absmax, blocksize=64):
    raise NotImplementedError
