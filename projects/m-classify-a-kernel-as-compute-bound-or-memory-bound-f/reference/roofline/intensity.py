def compute_intensity(flops: int, bytes_transferred: int) -> float:
    return float(flops) / float(bytes_transferred)
