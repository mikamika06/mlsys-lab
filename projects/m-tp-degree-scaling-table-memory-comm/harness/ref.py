import numpy as np

CONFIGS = [
    {"hidden_size": 1024, "intermediate_size": 4096, "num_layers": 4},
    {"hidden_size": 512, "intermediate_size": 2048, "num_layers": 2}
]

def compute_scaling_table(config, tp_degrees):
    out = []
    hs = config["hidden_size"]
    ffn = config["intermediate_size"]
    layers = config["num_layers"]
    for tp in tp_degrees:
        weight_mem = float(layers * (hs * ffn * 4 * 3) / (tp * 1024**3))
        comm_bytes = float(layers * 2 * hs * 4 * (tp - 1) / tp)
        out.append({"tp": int(tp), "memory_gb": weight_mem, "comm_bytes": comm_bytes})
    return out

def row_parallel_forward(weight, bias, x, rank, world_size):
    d_in = weight.shape[0]
    shard_size = d_in // world_size
    start = rank * shard_size
    end = start + shard_size
    w_shard = weight[start:end, :]
    x_shard = x[..., start:end]
    local_out = np.matmul(x_shard, w_shard)
    total_out = local_out * world_size
    if bias is not None:
        total_out += bias / world_size
    return total_out

def optimal_tp(config, hardware):
    table = compute_scaling_table(config, [1, 2, 4, 8])
    best = min(table, key=lambda d: d["memory_gb"] + d["comm_bytes"] / 1e8)
    return best["tp"]
