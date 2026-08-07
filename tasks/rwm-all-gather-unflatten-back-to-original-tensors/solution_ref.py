def unflatten_all_gathered(shards: list[list[float]], shapes: list[tuple[int, ...]]) -> list[list[float]]:
    flat_list = []
    for shard in shards:
        for val in shard:
            flat_list.append(val)

    total = 0
    for shape in shapes:
        prod = 1
        for dim in shape:
            prod = prod * dim
        total = total + prod

    flat = flat_list[:total] if flat_list else []

    params = []
    offset = 0
    for shape in shapes:
        size = 1
        for dim in shape:
            size = size * dim

        chunk = flat[offset:offset + size]

        def reshape_list(dims, flat_iter):
            if not dims:
                return next(flat_iter)
            dim_size = dims[0]
            rest = dims[1:]
            return [reshape_list(rest, flat_iter) for _ in range(dim_size)]

        it = iter(chunk)
        params.append(reshape_list(shape, it))
        offset += size
    return params
