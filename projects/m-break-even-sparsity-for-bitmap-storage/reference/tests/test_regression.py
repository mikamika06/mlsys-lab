import numpy as np
from edge_prune.storage import calculate_theoretical_size


def test_theoretical_size_includes_bitmap():
    mask = np.zeros(100, dtype=bool)
    mask[:50] = True
    masks = {"layer1": mask}
    size = calculate_theoretical_size(masks, 8)
    assert size == 63
