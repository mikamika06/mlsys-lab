import sys

sys.path.insert(0, ".")
from cacheverify.verify import verify_prefill_update
from cacheverify.metrics import peak_memory_delta


def test_verify_zero_error():
    x = [[1.0, 2.0], [3.0, 4.0]]
    assert verify_prefill_update(x, x) == 0.0


def test_peak_memory_positive():
    assert peak_memory_delta((1, 2, 3), 4) > 0.0
