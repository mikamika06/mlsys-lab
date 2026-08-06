import math


def get_dtensor_shard_info(global_shape, mesh_size, rank, shard_dim=0):
    if rank < 0 or rank >= mesh_size:
        raise ValueError("Invalid rank")

    dim_size = global_shape[shard_dim]
    base = dim_size // mesh_size
    rem = dim_size % mesh_size

    if rank < rem:
        local_dim = base + 1
        offset_dim = rank * (base + 1)
    else:
        local_dim = base
        offset_dim = rem * (base + 1) + (rank - rem) * base

    local_shape = list(global_shape)
    local_shape[shard_dim] = local_dim

    offset = [0] * len(global_shape)
    offset[shard_dim] = offset_dim

    numel_global = 1
    for s in global_shape:
        numel_global *= s

    numel_local = 1
    for s in local_shape:
        numel_local *= s

    return {
        "global_shape": tuple(global_shape),
        "local_shape": tuple(local_shape),
        "offset": tuple(offset),
        "numel_local": numel_local,
        "numel_global": numel_global,
    }


def compare_fsdp1_fsdp2_chunking(global_shape, mesh_size, dtype_bytes=4):
    dim0 = global_shape[0]
    trailing_numel = 1
    for s in global_shape[1:]:
        trailing_numel *= s

    fsdp1_local_dim0 = math.ceil(dim0 / mesh_size)
    fsdp1_padded_dim0 = fsdp1_local_dim0 * mesh_size

    fsdp1_padded_shape = (fsdp1_padded_dim0,) + tuple(global_shape[1:])
    fsdp1_local_shape = (fsdp1_local_dim0,) + tuple(global_shape[1:])

    fsdp1_total_numel = fsdp1_padded_dim0 * trailing_numel
    numel_global = dim0 * trailing_numel

    fsdp1_total_bytes = fsdp1_total_numel * dtype_bytes
    fsdp1_wasted_bytes = (fsdp1_total_numel - numel_global) * dtype_bytes

    base = dim0 // mesh_size
    rem = dim0 % mesh_size
    fsdp2_local_shapes = []
    for r in range(mesh_size):
        ld0 = base + 1 if r < rem else base
        fsdp2_local_shapes.append((ld0,) + tuple(global_shape[1:]))

    fsdp2_total_bytes = numel_global * dtype_bytes
    fsdp2_wasted_bytes = 0

    return {
        "fsdp1_padded_shape": tuple(fsdp1_padded_shape),
        "fsdp1_local_shape": tuple(fsdp1_local_shape),
        "fsdp1_total_bytes": fsdp1_total_bytes,
        "fsdp1_wasted_bytes": fsdp1_wasted_bytes,
        "fsdp2_local_shapes": fsdp2_local_shapes,
        "fsdp2_total_bytes": fsdp2_total_bytes,
        "fsdp2_wasted_bytes": fsdp2_wasted_bytes,
    }


def compute_reshard_memory_profile(
    layer_param_shapes, mesh_size, dtype_bytes=4, reshard_after_forward=True
):
    local_bytes_list = []
    full_bytes_list = []

    for shape in layer_param_shapes:
        numel_full = 1
        for s in shape:
            numel_full *= s
        full_b = numel_full * dtype_bytes

        shard_info = get_dtensor_shard_info(shape, mesh_size, 0, shard_dim=0)
        local_b = shard_info["numel_local"] * dtype_bytes

        local_bytes_list.append(local_b)
        full_bytes_list.append(full_b)

    total_sharded_bytes = sum(local_bytes_list)

    if reshard_after_forward:
        persistent_after_forward = total_sharded_bytes
        peak_during_forward = 0
        for i in range(len(layer_param_shapes)):
            curr = total_sharded_bytes - local_bytes_list[i] + full_bytes_list[i]
            if curr > peak_during_forward:
                peak_during_forward = curr
    else:
        persistent_after_forward = sum(full_bytes_list)
        peak_during_forward = 0
        accumulated = total_sharded_bytes
        for i in range(len(layer_param_shapes)):
            accumulated += full_bytes_list[i] - local_bytes_list[i]
            if accumulated > peak_during_forward:
                peak_during_forward = accumulated

    saved_bytes = (sum(full_bytes_list) - total_sharded_bytes) if reshard_after_forward else 0

    return {
        "persistent_param_bytes_after_forward": persistent_after_forward,
        "peak_param_bytes_during_forward": peak_during_forward,
        "saved_bytes_after_forward": saved_bytes,
    }


M1_CASES = [
    ((4096, 4096), 8, 0, 0),
    ((4096, 4096), 8, 3, 0),
    ((100, 512), 8, 0, 0),
    ((100, 512), 8, 3, 0),
    ((100, 512), 8, 7, 0),
    ((13, 128, 64), 4, 1, 0),
    ((13, 128, 64), 4, 2, 0),
    ((1000, 2048), 16, 5, 0),
    ((7, 32), 2, 0, 0),
    ((7, 32), 2, 1, 0),
]

M2_CASES = [
    ((100, 512), 8, 4),
    ((4096, 4096), 8, 2),
    ((1001, 1024), 16, 4),
    ((13, 128, 64), 4, 2),
    ((7, 32), 3, 4),
]

M3_CASES = [
    ([(100, 512), (200, 512), (100, 1024)], 8, 4, True),
    ([(100, 512), (200, 512), (100, 1024)], 8, 4, False),
    ([(4096, 4096)] * 4, 8, 2, True),
    ([(4096, 4096)] * 4, 8, 2, False),
]
