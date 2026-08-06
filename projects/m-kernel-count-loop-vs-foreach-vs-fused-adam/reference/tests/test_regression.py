import sys
import numpy as np

sys.path.insert(0, ".")
from opt.grouping import group_tensors_by_device_dtype
from opt.adam import step_loop, step_foreach, step_fused


def test_grouping_preserves_all_parameters():
    params = [
        {"id": 0, "device": "cuda:0", "dtype": "float32"},
        {"id": 1, "device": "cuda:0", "dtype": "float16"},
        {"id": 2, "device": "cuda:1", "dtype": "float32"},
        {"id": 3, "device": "cuda:0", "dtype": "float32"},
    ]
    groups = group_tensors_by_device_dtype(params)
    total_found = sum(len(v) for v in groups.values())
    assert total_found == len(params), f"Expected {len(params)} parameters, got {total_found}"


def test_fused_and_foreach_numeric_equivalence():
    np.random.seed(42)
    p1 = np.random.randn(10, 10).astype(np.float64)
    g1 = np.random.randn(10, 10).astype(np.float64)

    p2 = p1.copy()
    g2 = g1.copy()

    params1 = [{"param": p1, "grad": g1, "device": "cpu", "dtype": "float64"}]
    states1 = [{"exp_avg": np.zeros_like(p1), "exp_avg_sq": np.zeros_like(p1), "step": 0}]

    params2 = [{"param": p2, "grad": g2, "device": "cpu", "dtype": "float64"}]
    states2 = [{"exp_avg": np.zeros_like(p2), "exp_avg_sq": np.zeros_like(p2), "step": 0}]

    for _ in range(5):
        step_foreach(params1, states1, lr=1e-3)
        step_fused(params2, states2, lr=1e-3)

    assert np.allclose(p1, p2, atol=1e-12), "Foreach and fused results diverged!"
