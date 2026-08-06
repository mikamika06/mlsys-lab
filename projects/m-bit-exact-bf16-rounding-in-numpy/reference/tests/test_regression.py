import numpy as np
from bf16num.bf16 import fp32_to_bf16_bits
from bf16num.fp16 import fp16_subnormal_mask
from bf16num.ulp import compute_ulp, get_dtype_max


def test_bf16_rounding_rne():
    x = np.array([1.0 + 2.0**-7 + 2.0**-8], dtype=np.float32)
    bits = fp32_to_bf16_bits(x)
    assert bits[0] == 0x3F82


def test_fp16_subnormals():
    x = np.array([1e-5, 1.0], dtype=np.float32)
    mask = fp16_subnormal_mask(x)
    assert bool(mask[0]) is True
    assert not bool(mask[1])


def test_ulp_and_max():
    ulp_val = compute_ulp(np.array([1.0]), "fp16")[0]
    assert np.isclose(ulp_val, 2.0**-10)
    assert get_dtype_max("fp16") == 65504.0
