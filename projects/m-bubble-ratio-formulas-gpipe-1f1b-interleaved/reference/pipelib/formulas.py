def gpipe_bubble_ratio(p: int, m: int) -> float:
    """Compute theoretical bubble ratio for standard GPipe schedule."""
    if p <= 0 or m <= 0:
        raise ValueError("Pipeline stages and microbatches must be positive.")
    return (p - 1) / (m + p - 1)


def f1b_bubble_ratio(p: int, m: int) -> float:
    """Compute theoretical bubble ratio for standard 1F1B schedule."""
    if p <= 0 or m <= 0:
        raise ValueError("Pipeline stages and microbatches must be positive.")
    return (p - 1) / (m + p - 1)


def interleaved_1f1b_bubble_ratio(p: int, m: int, v: int) -> float:
    """Compute theoretical bubble ratio for interleaved 1F1B schedule."""
    if p <= 0 or m <= 0 or v <= 0:
        raise ValueError("Pipeline stages, microbatches, and virtual stages must be positive.")
    return (p - 1) / (v * m + p - 1)
