import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

class DummyAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(self, x):
        b, s, h = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, h)
        return self.out_proj(out)

def measure_block_flops(model, x):
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(x)
    return flop_counter.get_total_flops()

def compute_mfu(tokens_per_sec, param_count, peak_flops_per_sec):
    executed_flops_per_sec = tokens_per_sec * (6.0 * param_count)
    return executed_flops_per_sec / peak_flops_per_sec
