import numpy as np


def detect_inverted_scale(scale, reference_scale):
    """Detect inverted scale."""
    s = float(scale)
    ref = float(reference_scale)
    if s <= 0 or ref <= 0:
        return False
    dist_direct = abs(s - ref)
    dist_inverse = abs(s - (1.0 / ref))
    return dist_inverse < dist_direct
