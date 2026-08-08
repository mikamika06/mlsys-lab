def compute_arithmetic_intensity(flops, bytes_transferred):
    if bytes_transferred <= 0:
        return float('inf') if flops > 0 else 0.0
    return float(flops) / float(bytes_transferred)
