def activation_memory_bytes(b, s, h, a):
    return s * b * h * (34.0 + 5.0 * a * s / h)
