import torch
import torch.nn as nn
from ckptutils.exactness import verify_gradient_exactness as ref_exactness
from ckptutils.pareto import compute_pareto_curve as ref_pareto
from ckptutils.breakdown import analyze_op_breakdown as ref_breakdown


def get_test_setup():
    torch.manual_seed(1337)
    model = nn.ModuleList([
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 16)
    ])
    inputs = torch.randn(4, 16)
    strategies = [
        [False, False, False, False, False],
        [True, False, True, False, True],
        [True, True, True, True, True]
    ]
    return model, inputs, strategies
