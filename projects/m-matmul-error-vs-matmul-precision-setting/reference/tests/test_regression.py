import numpy as np
from numerics import matmul_chain

def test_tf32_is_more_precise_than_bf16():
    np.random.seed(42)
    matrices = [np.random.randn(16, 16).astype(np.float32) for _ in range(5)]

    fp32_res = matmul_chain(matrices, "fp32")
    tf32_res = matmul_chain(matrices, "tf32")
    bf16_res = matmul_chain(matrices, "bf16")

    err_tf32 = np.linalg.norm(tf32_res - fp32_res)
    err_bf16 = np.linalg.norm(bf16_res - fp32_res)

    assert err_tf32 < err_bf16, f"TF32 err {err_tf32} should be < BF16 err {err_bf16}"
