import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ref_mlp(x, w1, w2):
    h = torch.matmul(x, w1.t())
    h = F.gelu(h)
    out = torch.matmul(h, w2.t())
    return out


def compute_ref_comm(batch_size, seq_len, hidden_dim, ffn_dim, tp_size, element_size_bytes=2):
    if tp_size <= 1:
        return {
            "forward_all_reduce_bytes": 0,
            "backward_all_reduce_bytes": 0,
            "total_bytes_per_step": 0,
        }
    tokens = batch_size * seq_len
    ring_factor = 2 * ((tp_size - 1) / tp_size)
    fwd_ar = ring_factor * tokens * hidden_dim * element_size_bytes
    bwd_ar = ring_factor * tokens * hidden_dim * element_size_bytes
    return {
        "forward_all_reduce_bytes": int(fwd_ar),
        "backward_all_reduce_bytes": int(bwd_ar),
        "total_bytes_per_step": int(fwd_ar + bwd_ar),
    }
