import sys
import torch

sys.path.insert(0, ".")
from custom_op.rbf import rbf_interact

def test_autograd_correctness():
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
    gamma = 0.5

    torch.autograd.gradcheck(rbf_interact, (x, y, gamma), eps=1e-6, atol=1e-4)
