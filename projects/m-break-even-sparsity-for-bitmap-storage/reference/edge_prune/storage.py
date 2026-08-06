import math


def find_break_even_sparsity(bit_width):
    return 1.0 / bit_width


def calculate_theoretical_size(masks, bit_width):
    total_elements = 0
    nnz = 0
    for m in masks.values():
        total_elements += m.size
        nnz += int(m.sum())
    total_bits = total_elements + nnz * bit_width
    return math.ceil(total_bits / 8.0)
