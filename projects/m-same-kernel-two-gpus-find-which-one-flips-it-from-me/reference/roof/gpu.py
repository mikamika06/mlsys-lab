def find_flipping_gpu(gpus, kernel):
    total_bytes = kernel["bytes_read"] + kernel["bytes_written"]
    intensity = kernel["flops"] / total_bytes
    for gpu in gpus:
        ridge = gpu["peak_flop"] / gpu["peak_bw"]
        if intensity >= ridge:
            return gpu["name"]
    return None
