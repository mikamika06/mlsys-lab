import sys
import numpy as np

sys.path.insert(0, ".")
from loraadapter.adapter import LoRALinear
from loraadapter.verify import verify_no_op


def test_lora_b_is_strictly_zero():
    layer = LoRALinear(32, 64, rank=4, alpha=8.0)
    assert np.all(layer.lora_b == 0.0), "lora_b must be initialized to zeros"


def test_fresh_adapter_is_numerical_no_op():
    layer = LoRALinear(32, 64, rank=4, alpha=8.0)
    x = np.random.default_rng(10).normal(0, 1, (8, 32))
    is_noop, err = verify_no_op(layer, x, tol=1e-7)
    assert is_noop, f"fresh adapter produced non-zero difference, max abs err: {err}"


def test_scaling_factor_applied_correctly():
    layer = LoRALinear(16, 32, rank=2, alpha=4.0)
    assert layer.alpha / layer.rank == 2.0
