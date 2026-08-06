import torch
import torch.distributed as dist


def ulysses_all_to_all(
    tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group=None
) -> torch.Tensor:
    """Perform All-to-All tensor redistribution across sequence parallel rank group."""
    world_size = dist.get_world_size(group) if dist.is_initialized() else 1
    if world_size == 1:
        return tensor

    shape = list(tensor.shape)
    shape[scatter_dim] = shape[scatter_dim] // world_size
    scatter_strided = tensor.reshape(
        shape[:scatter_dim] + [world_size, shape[scatter_dim]] + shape[scatter_dim + 1 :]
    )

    perm = list(range(len(scatter_strided.shape)))
    perm = [0] + [scatter_dim + 1] + perm[1:scatter_dim + 1] + perm[scatter_dim + 2:]

    input_tensors = [
        t.contiguous() for t in torch.chunk(tensor, world_size, dim=scatter_dim)
    ]
    output_tensors = [torch.empty_like(input_tensors[0]) for _ in range(world_size)]
    dist.all_to_all(output_tensors, input_tensors, group=group)

    gathered = torch.cat(output_tensors, dim=gather_dim)
    return gathered


def ulysses_attention_forward(
    query_chunk: torch.Tensor,
    key_chunk: torch.Tensor,
    value_chunk: torch.Tensor,
    num_heads: int,
    group=None,
) -> torch.Tensor:
    """Compute sequence parallel attention using DeepSpeed-Ulysses pattern."""
    world_size = dist.get_world_size(group) if dist.is_initialized() else 1

    q_a2a = ulysses_all_to_all(query_chunk, scatter_dim=2, gather_dim=1, group=group)
    k_a2a = ulysses_all_to_all(key_chunk, scatter_dim=2, gather_dim=1, group=group)
    v_a2a = ulysses_all_to_all(value_chunk, scatter_dim=2, gather_dim=1, group=group)

    local_heads = num_heads // world_size
    head_dim = q_a2a.shape[-1] // num_heads

    q = q_a2a.view(q_a2a.shape[0], q_a2a.shape[1], local_heads, head_dim).transpose(1, 2)
    k = k_a2a.view(k_a2a.shape[0], k_a2a.shape[1], local_heads, head_dim).transpose(1, 2)
    v = v_a2a.view(v_a2a.shape[0], v_a2a.shape[1], local_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn_weights, v)

    context = context.transpose(1, 2).contiguous().view(context.shape[0], context.shape[2], -1)

    out = ulysses_all_to_all(context, scatter_dim=1, gather_dim=2, group=group)
    return out
