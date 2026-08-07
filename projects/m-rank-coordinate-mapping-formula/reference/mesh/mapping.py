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
