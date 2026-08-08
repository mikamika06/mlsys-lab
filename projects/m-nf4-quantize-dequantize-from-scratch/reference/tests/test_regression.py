import sys
import numpy as np

sys.path.insert(0, ".")
from nf4.codebooks import build_int4_codebook
from nf4.quantize import quantize_blockwise, dequantize_blockwise


def test_dequantize_uses_absmax():
    codebook = build_int4_codebook()
    tensor = np.ones(64, dtype=np.float32) * 5.0
    q, absmax = quantize_blockwise(tensor, codebook, block_size=64)
    deq = dequantize_blockwise(q, absmax, codebook, block_size=64)
    assert np.allclose(deq, tensor, atol=0.1), "Dequantized tensor does not scale by absmax!"
