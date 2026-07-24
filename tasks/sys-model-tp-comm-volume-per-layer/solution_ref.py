def tp_comm_volume_per_layer(b: int, s: int, h: int, N: int) -> float:
    return float(2 * b * s * h * ((N - 1) / N) * 2)
