import numpy as np


def get_test_configs():
    return [
        {
            "shapes": [(256, 256), (128, 128)],
            "world_sizes": [2, 4, 8],
            "total_bytes": 1024 * 1024,
            "bandwidth": 20.0
        },
        {
            "shapes": [(512, 512)],
            "world_sizes": [1, 3, 5],
            "total_bytes": 2 * 1024 * 1024,
            "bandwidth": 10.0
        },
        {
            "shapes": [(64, 64, 64)],
            "world_sizes": [2, 6],
            "total_bytes": 4 * 1024 * 1024,
            "bandwidth": 40.0
        }
    ]


def compute_oracle_file_sizes(shapes, world_size, dtype_bytes=4):
    total_elements = sum(int(np.prod(s)) for s in shapes)
    sizes = []
    for r in range(world_size):
        base = total_elements // world_size
        rem = total_elements % world_size
        cnt = base + (1 if r < rem else 0)
        sizes.append(cnt * dtype_bytes)
    return sizes
