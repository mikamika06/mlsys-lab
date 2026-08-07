import sys
sys.path.insert(0, ".")
from pipelp.throughput import compute_throughput
from pipelp.imbalance import find_imbalanced_stage
from pipelp.memory import peak_memory_1f1b, peak_memory_interleaved


def test_throughput_scaling():
    val = compute_throughput(4, 32, 1e15, 0.1)
    assert val > 0.0


def test_imbalance_detection():
    logs = [
        {"stage": 0, "activation_bytes": 100},
        {"stage": 1, "activation_bytes": 500},
        {"stage": 2, "activation_bytes": 150},
    ]
    assert find_imbalanced_stage(logs) == 1


def test_memory_bounds():
    m1 = peak_memory_1f1b(4, 16, 1024, 2048, 2)
    m_int = peak_memory_interleaved(4, 16, 4, 1024, 2048, 2)
    assert m_int < m1
