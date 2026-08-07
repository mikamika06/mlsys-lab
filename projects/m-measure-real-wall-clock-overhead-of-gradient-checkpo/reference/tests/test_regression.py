import torch
from chkpt.measure import measure_checkpoint_overhead
from chkpt.reentrant import run_reentrant_test
from chkpt.verify import count_recomputations

def test_measure_returns_float():
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())
    inputs = torch.randn(4, 16)
    t = measure_checkpoint_overhead(model, inputs)
    assert isinstance(t, float)
    assert t > 0.0

def test_reentrant_in_place_handling():
    model = torch.nn.Linear(16, 16)
    inputs = torch.randn(4, 16)
    success, err = run_reentrant_test(model, inputs, use_reentrant=False)
    assert success is True

def test_single_recomputation_count():
    model = torch.nn.Linear(16, 16)
    inputs = torch.randn(4, 16)
    count = count_recomputations(model, inputs)
    assert count == 2
