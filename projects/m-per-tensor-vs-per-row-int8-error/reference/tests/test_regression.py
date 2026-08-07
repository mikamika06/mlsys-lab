import torch
from quantutil.core import compute_error, select_config
from quantutil.config import apply_quantization


def test_compute_error_bounds():
    t = torch.ones(10, 10)
    err = compute_error(t, "per-tensor")
    assert err >= 0.0


def test_select_config_valid_output():
    cfg = select_config(0.001)
    assert cfg in ["per-tensor", "per-row"]


def test_apply_quantization_raises_without_compile():
    model = torch.nn.Linear(8, 8)
    try:
        apply_quantization(model, "per-row", use_compile=False)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
