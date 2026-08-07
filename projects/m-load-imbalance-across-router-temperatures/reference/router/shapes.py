import numpy as np


def verify_alltoall_shapes(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    world_size: int,
    capacity_factor: float,
    hidden_dim: int
) -> dict:
    experts_per_rank = num_experts // world_size
    tokens_per_expert = int(np.ceil((num_tokens * top_k / num_experts) * capacity_factor))
    dispatch_send_shape = (world_size, experts_per_rank * tokens_per_expert, hidden_dim)
    dispatch_recv_shape = (world_size, experts_per_rank * tokens_per_expert, hidden_dim)
    combine_send_shape = dispatch_recv_shape
    combine_recv_shape = dispatch_send_shape
    total_padded_tokens_per_rank = experts_per_rank * tokens_per_expert * world_size
    return {
        "dispatch_send_shape": dispatch_send_shape,
        "dispatch_recv_shape": dispatch_recv_shape,
        "combine_send_shape": combine_send_shape,
        "combine_recv_shape": combine_recv_shape,
        "tokens_per_expert": tokens_per_expert,
        "total_padded_tokens_per_rank": total_padded_tokens_per_rank
    }
