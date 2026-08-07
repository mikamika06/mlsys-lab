import numpy as np


def compute_superblock_footprint(num_elements, superblock_size, subblock_size, quant_bits, scale_bits, super_scale_bits):
    raise NotImplementedError


def quantize_superblock(data, superblock_size, subblock_size, quant_bits):
    raise NotImplementedError


def calculate_amortization_advantage(data, superblock_size, subblock_size, quant_bits):
    raise NotImplementedError
