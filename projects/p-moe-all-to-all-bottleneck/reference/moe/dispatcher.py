import numpy as np

def measure_exchange_volume(routes, world_size, num_experts, token_bytes):
    experts_per_rank = num_experts // world_size
    total_bytes = 0
    for r in range(world_size):
        rank_routes = routes[r].flatten()
        target_ranks = rank_routes // experts_per_rank
        remote_mask = (target_ranks != r)
        total_bytes += int(np.sum(remote_mask)) * token_bytes
    return total_bytes

def compute_imbalance_metrics(routes, world_size, num_experts, token_bytes):
    experts_per_rank = num_experts // world_size
    sent_bytes = np.zeros(world_size, dtype=int)
    recv_bytes = np.zeros(world_size, dtype=int)

    for r in range(world_size):
        rank_routes = routes[r].flatten()
        target_ranks = rank_routes // experts_per_rank
        for dst in target_ranks:
            if dst != r:
                sent_bytes[r] += token_bytes
                recv_bytes[dst] += token_bytes

    max_rank_bytes = int(np.max(np.maximum(sent_bytes, recv_bytes)))
    avg_rank_bytes = float(np.mean(np.maximum(sent_bytes, recv_bytes)))
    imbalance_factor = float(max_rank_bytes / (avg_rank_bytes + 1e-9))
    bottleneck_bytes = max_rank_bytes * world_size

    return {
        "max_rank_bytes": max_rank_bytes,
        "avg_rank_bytes": avg_rank_bytes,
        "imbalance_factor": imbalance_factor,
        "bottleneck_bytes": bottleneck_bytes
    }

def group_tokens_by_destination(tokens, routes, world_size, num_experts):
    experts_per_rank = num_experts // world_size
    grouped = {
        "local": [],
        "send_buffers": [],
        "send_counts": np.zeros((world_size, world_size), dtype=int)
    }

    for r in range(world_size):
        rank_tokens = tokens[r]
        rank_routes = routes[r]
        num_tokens, top_k = rank_routes.shape

        local_toks = []
        remote_toks = []
        remote_dsts = []

        for i in range(num_tokens):
            for k in range(top_k):
                exp_id = rank_routes[i, k]
                dst_rank = exp_id // experts_per_rank
                tok = rank_tokens[i]
                if dst_rank == r:
                    local_toks.append((tok, exp_id))
                else:
                    remote_toks.append((tok, exp_id))
                    remote_dsts.append(dst_rank)
                    grouped["send_counts"][r, dst_rank] += 1

        grouped["local"].append(local_toks)

        if remote_toks:
            sort_idx = np.argsort(remote_dsts)
            sorted_remote = np.array([remote_toks[idx][0] for idx in sort_idx])
            grouped["send_buffers"].append(sorted_remote)
        else:
            grouped["send_buffers"].append(np.empty((0, tokens.shape[2])))

    return grouped

def overlap_compute_and_comm(tokens, routes, world_size, num_experts, compute_cost_per_token=1.0, comm_cost_per_byte=0.01):
    experts_per_rank = num_experts // world_size
    token_bytes = tokens.shape[2] * tokens.itemsize
    grouped = group_tokens_by_destination(tokens, routes, world_size, num_experts)

    max_local_time = 0.0
    max_comm_time = 0.0
    max_remote_compute_time = 0.0

    for r in range(world_size):
        num_local = len(grouped["local"][r])
        local_time = num_local * compute_cost_per_token
        max_local_time = max(max_local_time, local_time)

        sent_bytes = np.sum(grouped["send_counts"][r]) * token_bytes
        recv_bytes = np.sum(grouped["send_counts"][:, r]) * token_bytes
        comm_time = max(sent_bytes, recv_bytes) * comm_cost_per_byte
        max_comm_time = max(max_comm_time, comm_time)

        num_received = np.sum(grouped["send_counts"][:, r])
        remote_compute_time = num_received * compute_cost_per_token
        max_remote_compute_time = max(max_remote_compute_time, remote_compute_time)

    sequential_time = max_local_time + max_comm_time + max_remote_compute_time
    overlapped_time = max(max_local_time, max_comm_time) + max_remote_compute_time

    return {
        "sequential_time": float(sequential_time),
        "overlapped_time": float(overlapped_time),
        "time_saved": float(sequential_time - overlapped_time)
    }

def optimize_and_evaluate_exchange(tokens, routes, world_size, num_experts, token_bytes):
    experts_per_rank = num_experts // world_size
    total_routing_tokens = routes.size
    total_bytes = total_routing_tokens * token_bytes

    comm_bytes = measure_exchange_volume(routes, world_size, num_experts, token_bytes)
    naive_comm_share = comm_bytes / float(total_bytes)

    time_stats = overlap_compute_and_comm(tokens, routes, world_size, num_experts)
    seq_time = time_stats["sequential_time"]
    ovl_time = time_stats["overlapped_time"]

    comm_overhead_overlapped = max(0.0, seq_time - ovl_time)
    optimized_comm_share = comm_overhead_overlapped / (ovl_time + 1e-9)

    return {
        "naive_comm_share": float(naive_comm_share),
        "optimized_comm_share": float(optimized_comm_share),
        "comm_volume_reduction": float(1.0 - comm_bytes / float(total_bytes))
    }

def dispatch_and_combine(tokens, routes, world_size, num_experts, expert_weights):
    num_ranks, num_tokens, hidden_dim = tokens.shape
    top_k = routes.shape[2]

    output_tokens = np.zeros_like(tokens)

    for r in range(num_ranks):
        for i in range(num_tokens):
            combined_tok = np.zeros(hidden_dim, dtype=tokens.dtype)
            for k in range(top_k):
                exp_id = routes[r, i, k]
                weight = expert_weights[exp_id]
                processed = tokens[r, i] * weight
                combined_tok += processed
            output_tokens[r, i] = combined_tok

    return output_tokens
