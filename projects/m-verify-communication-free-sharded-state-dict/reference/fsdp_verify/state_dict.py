"""Verify communication-free sharded state dict placements."""


def verify_communication_free_state_dict(param_specs, world_size, rank):
    """Check if all parameter shards on this rank are local and communication-free."""
    is_comm_free = True
    comm_bytes = 0
    param_status = {}

    for name, spec in param_specs.items():
        shape = spec["shape"]
        shard_dim = spec.get("shard_dim", 0)
        placements = spec.get("placements", {})

        total_elements = 1
        for d in shape:
            total_elements *= d

        dim_size = shape[shard_dim] if len(shape) > shard_dim else 1
        shard_size = (dim_size + world_size - 1) // world_size

        expected_start = rank * shard_size
        expected_end = min((rank + 1) * shard_size, dim_size)
        expected_len = max(0, expected_end - expected_start)

        rank_placement = placements.get(rank, {})
        actual_start = rank_placement.get("start", -1)
        actual_end = rank_placement.get("end", -1)
        requires_comm = rank_placement.get("requires_comm", False)

        mismatch = (actual_start != expected_start) or (actual_end != expected_end) or requires_comm

        if mismatch:
            is_comm_free = False
            elem_size = spec.get("dtype_bytes", 4)
            missing = abs(expected_len - max(0, actual_end - actual_start))
            if missing == 0 or requires_comm:
                missing = expected_len
            bytes_needed = missing * (total_elements // max(1, dim_size)) * elem_size
            comm_bytes += bytes_needed
            param_status[name] = {"comm_free": False, "comm_bytes": bytes_needed}
        else:
            param_status[name] = {"comm_free": True, "comm_bytes": 0}

    return {
        "is_communication_free": is_comm_free,
        "total_comm_bytes": comm_bytes,
        "parameters": param_status,
    }
