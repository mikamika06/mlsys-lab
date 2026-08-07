def reduce_scatter_owner(grads: list[list[float]], world_size: int) -> list[list[float]]:
    num_grads = len(grads)
    grad_len = len(grads[0])
    reduced = [0.0] * grad_len
    for i in range(grad_len):
        col_sum = 0.0
        for r in range(num_grads):
            col_sum += grads[r][i]
        reduced[i] = col_sum

    shard_size = grad_len // world_size
    return [
        reduced[r * shard_size:(r + 1) * shard_size]
        for r in range(world_size)
    ]
