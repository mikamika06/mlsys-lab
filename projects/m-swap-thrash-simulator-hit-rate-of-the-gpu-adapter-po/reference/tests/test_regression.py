import sys
import numpy as np

sys.path.insert(0, ".")
from multilora.simulator import simulate_hit_rate
from multilora.validator import validate_adapters
from multilora.lora import lora_forward


def test_simulator_basic():
    rate = simulate_hit_rate([1, 1, 1], 1)
    assert rate == 2.0 / 3.0


def test_validator_raises_on_overflow():
    adapters = [{"rank": 16, "memory_mb": 600}]
    limits = {"max_rank": 32, "max_memory_mb": 500}
    try:
        validate_adapters(adapters, limits)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_lora_forward_math():
    x = np.ones((1, 4))
    base_w = np.zeros((4, 4))
    la = np.ones((4, 2))
    lb = np.ones((2, 4))
    res = lora_forward(base_w, la, lb, 1.0, x)
    assert res.shape == (1, 4)
