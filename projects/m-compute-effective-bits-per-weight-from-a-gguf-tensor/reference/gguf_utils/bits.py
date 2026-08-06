def compute_effective_bits(tensors):
    total_bits = 0
    total_weights = 0
    for t in tensors:
        n_elements = t["n_elements"]
        bpw = t["bits_per_weight"]
        total_bits += n_elements * bpw
        total_weights += n_elements
    if total_weights == 0:
        return 0.0
    return float(total_bits) / float(total_weights)
