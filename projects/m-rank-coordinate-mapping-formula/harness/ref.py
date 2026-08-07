import numpy as np

MESH_SHAPES = [(2, 4, 8), (4, 4, 2), (2, 2, 2, 2), (8, 16)]


def rank_to_coords(rank, mesh_shape):
    coords = []
    for dim in reversed(mesh_shape):
        coords.insert(0, rank % dim)
        rank //= dim
    return tuple(coords)


def coords_to_rank(coords, mesh_shape):
    rank = 0
    for c, dim in zip(coords, mesh_shape):
        rank = rank * dim + c
    return rank


def get_subgroup_ranks(mesh_shape, fixed_axes):
    total = int(np.prod(mesh_shape))
    ranks = []
    for r in range(total):
        coords = rank_to_coords(r, mesh_shape)
        match = True
        for ax, val in fixed_axes.items():
            if coords[ax] != val:
                match = False
                break
        if match:
            ranks.append(r)
    return sorted(ranks)


def reconstruct_mesh_shape(ranks):
    total = len(ranks)
    for shape in MESH_SHAPES:
        if int(np.prod(shape)) == total:
            return shape
    return (total,)
