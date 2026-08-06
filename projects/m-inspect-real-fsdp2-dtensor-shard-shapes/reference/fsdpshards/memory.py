from fsdpshards.sharding import get_dtensor_shard_info


def compute_reshard_memory_profile(
    layer_param_shapes: list[tuple[int, ...]],
    mesh_size: int,
    dtype_bytes: int = 4,
    reshard_after_forward: bool = True,
) -> dict:
    """Measure param memory footprint during and after forward for reshard settings."""
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
