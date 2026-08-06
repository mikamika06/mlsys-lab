import sys
import numpy as np

sys.path.insert(0, ".")
from layout.chooser import check_memory_fit, select_layout
from layout.imbalance import straggler_factor

CONFIG = {
    "num_layers": 32,
    "weight_bytes": 16 * (1024 ** 3),
    "kv_bytes_per_token_per_layer": 4096,
    "activation_bytes_per_token_per_layer": 8192,
    "max_seq_len": 2048,
    "batch_size": 4,
}


def test_memory_fit_bounds():
    assert check_memory_fit(CONFIG, tp=8, pp=1, vram_gb=16.0) is True
    assert check_memory_fit(CONFIG, tp=1, pp=1, vram_gb=2.0) is False


def test_select_layout_respects_vram():
    latency_table = {
        (1, 1, 8): 10.0,
        (2, 1, 4): 20.0,
        (4, 1, 2): 30.0,
        (8, 1, 1): 40.0,
        (1, 2, 4): 15.0,
        (2, 2, 2): 25.0,
        (4, 2, 1): 35.0,
        (1, 4, 2): 18.0,
        (2, 4, 1): 28.0,
        (1, 8, 1): 38.0,
    }
    idx = select_layout(CONFIG, vram_gb=10.0, latency_table=latency_table)
    assert idx != 0
    assert idx != -1


def test_straggler_factor_balanced():
    hist = np.array([100, 100, 100, 100])
    assert abs(straggler_factor(hist) - 1.0) < 1e-6
