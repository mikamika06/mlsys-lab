def classify_kernel(bytes_read, bytes_written, flops, ridge_point):
    total_bytes = bytes_read + bytes_written
    intensity = flops / total_bytes if total_bytes > 0 else 0.0
    return "compute" if intensity >= ridge_point else "memory"
