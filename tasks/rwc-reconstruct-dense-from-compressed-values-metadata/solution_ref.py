import numpy as np


def reconstruct_dense(values, metadata, shape):
    values = np.asarray(values, dtype=np.float64)
    metadata = np.asarray(metadata, dtype=np.uint8)

    dense = np.zeros(int(np.prod(shape)), dtype=np.float64)
    positions = np.empty(metadata.size * 2, dtype=np.int64)

    value_index = 0
    pos_index = 0

    for block, code in enumerate(metadata):
        p0 = int(code & 3)
        p1 = int((code >> 2) & 3)

        base = block * 4
        positions[pos_index] = p0
        positions[pos_index + 1] = p1

        dense[base + p0] = values[value_index]
        dense[base + p1] = values[value_index + 1]

        value_index += 2
        pos_index += 2

    return dense.reshape(shape), positions
