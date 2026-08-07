import numpy as np


def reconstruct_mesh_shape(ranks):
    total = len(ranks)
    candidates = [(2, 4, 8), (4, 4, 2), (2, 2, 2, 2), (8, 16)]
    for shape in candidates:
        if int(np.prod(shape)) == total:
            return shape
    return (total,)
