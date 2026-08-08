CONFIGS = [
    {"num_params": 1000, "world_size": 4},
    {"num_params": 1000003, "world_size": 8},
    {"num_params": 7000000000, "world_size": 64},
]

COMM_CONFIGS = [
    {"num_params": 1000000, "world_size": 8, "bytes_per_param": 2, "bytes_per_grad": 2},
    {"num_params": 7000000000, "world_size": 64, "bytes_per_param": 2, "bytes_per_grad": 4},
    {"num_params": 50000000, "world_size": 16, "bytes_per_param": 4, "bytes_per_grad": 4},
]

MEMORY_CONFIGS = [
    {
        "layer_params": 10000000,
        "world_size": 8,
        "rank": 0,
        "batch_size": 4,
        "seq_len": 2048,
        "hidden_dim": 4096,
        "bytes_per_param": 2,
        "bytes_per_activation": 2,
    },
    {
        "layer_params": 10000005,
        "world_size": 8,
        "rank": 7,
        "batch_size": 2,
        "seq_len": 4096,
        "hidden_dim": 4096,
        "bytes_per_param": 2,
        "bytes_per_activation": 2,
    },
]


def ref_compute_rank_shard_size(num_params: int, world_size: int, rank: int) -> int:
    base = num_params // world_size
    remainder = num_params % world_size
    return base + 1 if rank < remainder else base


def ref_compute_world_shard_distribution(num_params: int, world_size: int) -> list[int]:
    return [ref_compute_rank_shard_size(num_params, world_size, r) for r in range(world_size)]


def ref_compute_per_step_communication_bytes(
    num_params: int,
    world_size: int,
    sharding_strategy: str,
    bytes_per_param: int = 2,
    bytes_per_grad: int = 2,
) -> int:
    if world_size <= 1:
        return 0
    scale = (world_size - 1) / world_size
    if sharding_strategy == "FULL_SHARD":
        param_bytes = num_params * bytes_per_param
        grad_bytes = num_params * bytes_per_grad
        return int(round(2 * scale * param_bytes + scale * grad_bytes))
    elif sharding_strategy == "SHARD_GRAD_OP":
        param_bytes = num_params * bytes_per_param
        grad_bytes = num_params * bytes_per_grad
        return int(round(scale * param_bytes + scale * grad_bytes))
    elif sharding_strategy == "NO_SHARD":
        grad_bytes = num_params * bytes_per_grad
        return int(round(scale * grad_bytes))
    else:
        raise ValueError(f"Unknown strategy: {sharding_strategy}")


def ref_compute_layer_transient_peak_memory_bytes(
    layer_params: int,
    world_size: int,
    rank: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    bytes_per_param: int = 2,
    bytes_per_activation: int = 2,
) -> int:
    sharded_param_bytes = ref_compute_rank_shard_size(layer_params, world_size, rank) * bytes_per_param
    full_param_bytes = layer_params * bytes_per_param
    input_act_bytes = batch_size * seq_len * hidden_dim * bytes_per_activation
    output_act_bytes = batch_size * seq_len * hidden_dim * bytes_per_activation

    return sharded_param_bytes + full_param_bytes + input_act_bytes + output_act_bytes
