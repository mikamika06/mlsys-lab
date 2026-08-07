import math

CONFIGS = [
    {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype_bytes": 2,
        "block_size": 16,
    },
    {
        "num_layers": 80,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype_bytes": 2,
        "block_size": 32,
    },
    {
        "num_layers": 48,
        "num_kv_heads": 4,
        "head_dim": 64,
        "dtype_bytes": 1,
        "block_size": 16,
    },
]

WORKLOADS = [
    [100, 250, 512, 1024],
    [1, 15, 16, 17, 33],
    [4096, 2048, 1024, 512],
]

PREEMPTION_EVENTS = [
    {"type": "swap_out", "seq_id": 1, "tokens": 100},
    {"type": "swap_out", "seq_id": 2, "tokens": 250},
    {"type": "swap_in", "seq_id": 1, "tokens": 120},
    {"type": "swap_out", "seq_id": 3, "tokens": 512},
    {"type": "swap_in", "seq_id": 2, "tokens": 250},
]


def block_bytes(config):
    return (
        2
        * config["num_layers"]
        * config["num_kv_heads"]
        * config["head_dim"]
        * config["block_size"]
        * config["dtype_bytes"]
    )


def sequence_blocks(token_count, block_size):
    return math.ceil(token_count / block_size)


def compute_sequence_swap_bytes(config, token_count):
    blk = block_bytes(config)
    num_blocks = sequence_blocks(token_count, config["block_size"])
    return num_blocks * blk


def compute_total_swap_bytes(config, token_counts):
    return sum(compute_sequence_swap_bytes(config, t) for t in token_counts)


def simulate_preemption_trajectory(config, events):
    blk = block_bytes(config)
    allocated_blocks = 0
    active_seqs = {}
    trajectory = []
    peak_bytes = 0

    for ev in events:
        etype = ev["type"]
        sid = ev["seq_id"]
        tokens = ev["tokens"]
        if etype == "swap_out":
            n_blocks = sequence_blocks(tokens, config["block_size"])
            active_seqs[sid] = n_blocks
            allocated_blocks += n_blocks
        elif etype == "swap_in":
            if sid in active_seqs:
                allocated_blocks -= active_seqs.pop(sid)

        curr_bytes = allocated_blocks * blk
        trajectory.append(curr_bytes)
        if curr_bytes > peak_bytes:
            peak_bytes = curr_bytes

    return {"trajectory": trajectory, "peak_bytes": peak_bytes}
