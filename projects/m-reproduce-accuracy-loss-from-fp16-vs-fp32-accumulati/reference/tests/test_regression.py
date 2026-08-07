import sys
sys.path.insert(0, ".")
from matmul_acc.kernel import compute_matmul_accumulation
from matmul_acc.scheduling import compute_l2_hit_rate
import numpy as np


def test_accumulation_precision():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((128, 512), dtype=np.float32)
    b = rng.standard_normal((512, 128), dtype=np.float32)
    res_fp32 = compute_matmul_accumulation(a, b, use_fp32_acc=True)
    res_fp16 = compute_matmul_accumulation(a, b, use_fp32_acc=False)
    ref = np.dot(a, b)
    err_fp32 = np.linalg.norm(res_fp32 - ref) / (np.linalg.norm(ref) + 1e-7)
    err_fp16 = np.linalg.norm(res_fp16 - ref) / (np.linalg.norm(ref) + 1e-7)
    assert err_fp32 < err_fp16
    assert err_fp32 < 1e-3


def test_scheduling_hit_rate():
    rate_row = compute_l2_hit_rate(1024, 1024, 1024, 64, 64, grouped=False)
    rate_group = compute_l2_hit_rate(1024, 1024, 1024, 64, 64, grouped=True)
    assert rate_group > rate_row
