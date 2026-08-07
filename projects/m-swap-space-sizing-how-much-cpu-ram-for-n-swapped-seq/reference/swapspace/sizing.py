import math


def compute_block_bytes(config):
    return (
        2
        * config["num_layers"]
        * config["num_kv_heads"]
        * config["head_dim"]
        * config["block_size"]
        * config["dtype_bytes"]
    )


def compute_sequence_swap_bytes(config, token_count):
    blk = compute_block_bytes(config)
    num_blocks = math.ceil(token_count / config["block_size"])
    return num_blocks * blk


def compute_total_swap_bytes(config, token_counts):
    return sum(compute_sequence_swap_bytes(config, t) for t in token_counts)
