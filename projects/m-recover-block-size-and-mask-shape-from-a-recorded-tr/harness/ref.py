import random
import numpy as np


def generate_trace(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    events = []
    base_offset = random.randint(100, 1000)
    stride = random.choice([4, 8, 16])
    for i in range(64):
        events.append({"offset": base_offset + i * stride})

    h, w = 16, 32
    mask_matrix = np.zeros((h, w), dtype=bool)
    active_h = random.randint(8, h)
    active_w = random.randint(16, w)
    mask_matrix[:active_h, :active_w] = True

    for row in mask_matrix:
        events.append({"mask": row.tolist()})

    return events, (16, w), (active_h, active_w)


TEST_CASES = [generate_trace(i) for i in range(10)]
