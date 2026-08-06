def compute_effective_bpw(tensors):
    total_bits = 0
    total_weights = 0
    for t in tensors:
        nelements = t["nelements"]
        bits_per_elem = t["bits_per_elem"]
        total_bits += nelements * bits_per_elem
        total_weights += nelements
    if total_weights == 0:
        return 0.0
    return total_bits / total_weights
