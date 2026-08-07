import numpy as np
from q4k.quantize import quantize_q4_k, dequantize_q4_k
from q4k.analysis import dominant_subblock
from q4k.compare import compare_q4_k_q4_0


def get_test_tensor():
    rng = np.random.default_rng(12345)
    return rng.standard_normal(512).astype(np.float32)
