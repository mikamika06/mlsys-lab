import sys
sys.path.insert(0, ".")
from tpscaling.table import compute_scaling_table
from tpscaling.rowparallel import row_parallel_forward
from tpscaling.sweetspot import optimal_tp
import numpy as np

def test_optimal_tp_is_valid():
    cfg = {"hidden_size": 1024, "intermediate_size": 4096, "num_layers": 4}
    tp = optimal_tp(cfg, {})
    assert tp in [1, 2, 4, 8], f"invalid optimal tp {tp}"

def test_table_non_empty():
    cfg = {"hidden_size": 1024, "intermediate_size": 4096, "num_layers": 4}
    t = compute_scaling_table(cfg, [1, 2])
    assert len(t) == 2

def test_row_parallel_shape():
    np.random.seed(0)
    w = np.random.randn(512, 256)
    b = np.random.randn(256)
    x = np.random.randn(2, 512)
    res = row_parallel_forward(w, b, x, 0, 2)
    assert res.shape == (2, 256)
