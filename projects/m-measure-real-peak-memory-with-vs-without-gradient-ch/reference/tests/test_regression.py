import sys
import torch
sys.path.insert(0, ".")
from gradckpt.memory import measure_peak_memory
from gradckpt.flow import check_gradient_flow_frozen
import ref

def test_memory_reduction():
    model = ref.ToyModel(hidden_dim=16, num_layers=4)
    x = torch.randn(2, 16)
    res = measure_peak_memory(model, x)
    assert res["mem_with"] < res["mem_without"]

def test_frozen_gradient_failure():
    model = ref.ToyModel(hidden_dim=16, num_layers=4)
    x = torch.randn(2, 16)
    failed = check_gradient_flow_frozen(model, x)
    assert failed is True

def test_fix_restores_flow():
    model = ref.ToyModel(hidden_dim=16, num_layers=4)
    x = torch.randn(2, 16)
    from gradckpt.fix import verify_input_require_grads_fix
    res = verify_input_require_grads_fix(model, x)
    assert res["has_grad"] is True
