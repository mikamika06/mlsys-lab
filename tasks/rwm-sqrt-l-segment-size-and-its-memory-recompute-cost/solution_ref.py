import math


def checkpoint_cost(L: int) -> tuple[int, int, int]:
    segment_size = int(round(math.sqrt(L)))
    stored_activations = 2 * segment_size
    extra_forward = L
    return segment_size, stored_activations, extra_forward
