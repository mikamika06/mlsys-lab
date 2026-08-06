def calculate_bytes(shape, dtype_size, num_ops, fused):
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    if fused:
        return num_elements * dtype_size * 2
    else:
        return num_elements * dtype_size * (num_ops + 1)
