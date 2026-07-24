def reduce_scatter_owner(grads, world_size):
    # TODO: fixes the wrong-owner bug by reducing after ownership routing.
    # This broken version assigns each rank its local shard instead of the
    # globally reduced shard that it owns.
    size = len(grads[0]) // world_size
    return [
        list(grads[r][r * size:(r + 1) * size])
        for r in range(world_size)
    ]
