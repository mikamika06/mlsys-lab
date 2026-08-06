import sys
sys.path.insert(0, ".")
from sparse.masks import generate_nm_mask
from sparse.bench import parse_a100_gemm_log
import numpy as np


def test_mask_generation_shape_and_sparsity():
    arr = np.ones((4, 8))
    mask = generate_nm_mask(arr, 2, 4)
    assert mask.shape == (4, 8)
    reshaped = mask.reshape(-1, 4)
    for block in reshaped:
        assert np.sum(block) == 2


def test_parse_a100_gemm_log():
    log = "kernel: test | dense_tflops: 100.0 TFLOPS | speedup: 2.0x"
    res = parse_a100_gemm_log(log)
    assert len(res) > 0
    item = res[0]
    assert item is not None
