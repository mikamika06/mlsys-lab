import torch
import torch.nn as nn
from dyncomp.backends import find_experimental_backends as ref_find_experimental_backends
from dyncomp.noop import noop_backend as ref_noop_backend
from dyncomp.metrics import measure_ratios as ref_measure_ratios


def get_test_model():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
    inputs = [torch.randn(4, 16)]
    return model, inputs
