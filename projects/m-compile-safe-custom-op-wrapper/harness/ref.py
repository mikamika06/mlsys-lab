import torch

def get_test_inputs():
    torch.manual_seed(42)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    return q, k, v

def expected_attention(q, k, v):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)
