def compare_size_ratios(tensor_shapes, quants):
    base_tensors = [{"nelements": s, "bits_per_elem": 32.0} for s in tensor_shapes]
    base_bits = sum(t["nelements"] * t["bits_per_elem"] for t in base_tensors)
    ratios = {}
    for q_name, tensors in quants.items():
        q_bits = sum(t["nelements"] * t["bits_per_elem"] for t in tensors)
        ratios[q_name] = q_bits / base_bits
    return ratios
