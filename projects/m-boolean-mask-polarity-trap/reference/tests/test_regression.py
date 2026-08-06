import torch
from sdpa_trap.attention import compute_manual, compute_sdpa

def test_attention_match():
    torch.manual_seed(100)
    B, H, S, D = 2, 2, 8, 8
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    pad_mask = torch.zeros(B, S, dtype=torch.bool)
    pad_mask[:, 4:] = True

    out_manual = compute_manual(q, k, v, pad_mask)
    out_sdpa = compute_sdpa(q, k, v, pad_mask)

    assert torch.allclose(out_manual, out_sdpa, atol=1e-4)

def test_sdpa_no_nan_on_unpadded():
    torch.manual_seed(101)
    B, H, S, D = 1, 1, 4, 4
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    pad_mask = torch.zeros(B, S, dtype=torch.bool)
    
    out = compute_sdpa(q, k, v, pad_mask)
    assert not torch.isnan(out).any()
