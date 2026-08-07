import sys
import torch

sys.path.insert(0, ".")
from scaler_lab.run_loop import run_scaling_loop
from scaler_lab.skipped_step import verify_skipped_step
from scaler_lab.compare_scales import compare_fp16_bf16

def test_scaling_loop_executes():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stream = [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(5)]
    scales = run_scaling_loop(model, opt, stream, inject_inf_steps=[2])
    assert len(scales) == 5

def test_skipped_step_leaves_weights_unchanged():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 4)
    res = verify_skipped_step(model, opt, x)
    assert res is True

def test_compare_scales_output():
    model_fn = lambda: torch.nn.Linear(4, 2)
    stream = [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(3)]
    res = compare_fp16_bf16(model_fn, stream, inject_inf_steps=[1])
    assert "scales_fp16" in res
    assert "loss_bf16" in res
