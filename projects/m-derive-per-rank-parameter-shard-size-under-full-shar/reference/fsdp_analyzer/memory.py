from fsdp_analyzer.sharding import compute_rank_shard_size


def compute_layer_transient_peak_memory_bytes(
    layer_params: int,
    world_size: int,
    rank: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    bytes_per_param: int = 2,
    bytes_per_activation: int = 2,
) -> int:
    sharded_param_bytes = compute_rank_shard_size(layer_params, world_size, rank) * bytes_per_param
    full_param_bytes = layer_params * bytes_per_param
    input_act_bytes = batch_size * seq_len * hidden_dim * bytes_per_activation
    output_act_bytes = batch_size * seq_len * hidden_dim * bytes_per_activation

    return sharded_param_bytes + full_param_bytes + input_act_bytes + output_act_bytes
