import numpy as np


def moe_all_to_all(send: np.ndarray, world_size: int) -> np.ndarray:
    received = []
    for dst in range(world_size):
        blocks = []
        for src in range(world_size):
            blocks.append(send[src, dst])
        received.append(np.concatenate(blocks, axis=0))
    return np.stack(received, axis=0)
