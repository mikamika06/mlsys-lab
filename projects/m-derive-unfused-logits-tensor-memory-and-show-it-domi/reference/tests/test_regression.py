import sys
import torch

sys.path.insert(0, ".")
from logits.memory import unfused_logits_bytes, weight_memory_bytes, logits_dominates_weights
from logits.chunked import chunked_crossentropy_bytes, memory_savings_ratio
from logits.kernel import fused_logits_forward_ref, fused_logits_forward_triton


def test_memory_scaling():
    b = unfused_logits_bytes(2, 4096, 32000, 2)
    assert b > 0


def test_chunked_savings():
    ratio = memory_savings_ratio(2, 4096, 32000, 1024, 2)
    assert ratio > 1.0


def test_kernel_equivalence():
    x = torch.randn(4, 16)
    w = torch.randn(32, 16)
    out_ref = fused_logits_forward_ref(x, w)
    out_tri = fused_logits_forward_triton(x, w)
    assert torch.allclose(out_ref, out_tri, atol=1e-5)
