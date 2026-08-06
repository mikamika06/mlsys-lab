import sys
sys.path.insert(0, ".")
import numpy as np
from iqquant.dequant import dequantize_iq4_nl
from iqquant.superblocks import decode_iq4_xs
from iqquant.bpw import compute_bpw

def test_iq4_nl_bounds():
    data = b"\x00\xff"
    scales = np.array([1.0], dtype=np.float32)
    out = dequantize_iq4_nl(data, scales)
    assert out.size == 4
    assert np.all(np.abs(out) <= 2.0)

def test_iq4_xs_decoding():
    block = bytes([100] + [128] * 32)
    out = decode_iq4_xs(block)
    assert out.size == 32
    assert np.all(out == 0.0)

def test_bpw_values():
    for t in ["IQ1_S", "IQ2_XXS", "IQ4_XS", "TQ1_0", "TQ2_0"]:
        bpw = compute_bpw(t)
        assert bpw > 0.0
