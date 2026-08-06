def compute_storage_bytes(num_elements, bits, block_size):
    raise NotImplementedError


def compute_mse(original, dequantized):
    raise NotImplementedError


def measure_tradeoff(tensor, block_sizes, bits_list):
    raise NotImplementedError
