import numpy as np


def toy_reduce_scatter(gradients, world_size):
    grads = np.array(gradients, dtype=np.float32)
    total_elements = grads.size
    remainder = total_elements % world_size
    if remainder != 0:
        pad_size = world_size - remainder
        grads = np.pad(grads, (0, pad_size), mode='constant')

    chunk_size = grads.size // world_size
    chunks = np.split(grads, world_size)

    reduced_chunks = []
    for i in range(world_size):
        reduced_chunks.append(float(np.sum([chunks[j][i] for j in range(world_size)])))
    return reduced_chunks
