from roofline.intensity import compute_intensity


def classify_kernel(flops: int, bytes_transferred: int, peak_flops: float, peak_bandwidth: float) -> tuple:
    intensity = compute_intensity(flops, bytes_transferred)
    crossover = peak_flops / peak_bandwidth
    if intensity < crossover:
        return "memory-bound", intensity, crossover
    return "compute-bound", intensity, crossover
