import sys
sys.path.insert(0, ".")
from triton_bw.kernels import atomic_heavy_backward, disjoint_write_backward
import torch


def test_atomic_and_disjoint_equivalence():
    x = torch.ones(8, 8)
    g = torch.ones(8, 8)
    r1 = atomic_heavy_backward(x, g)
    r2 = disjoint_write_backward(x, g)
    assert torch.allclose(r1, r2, atol=1e-5, rtol=1e-5)
