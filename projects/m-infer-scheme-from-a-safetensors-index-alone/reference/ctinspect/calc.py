import math

def calculate_quant_byte_size(shape, num_bits, packed):
    total_elements = 1
    for d in shape:
        total_elements *= d

    if packed:
        elements_per_byte = 8 // num_bits
        return math.ceil(total_elements / elements_per_byte)
    else:
        bytes_per_element = math.ceil(num_bits / 8)
        return total_elements * bytes_per_element
