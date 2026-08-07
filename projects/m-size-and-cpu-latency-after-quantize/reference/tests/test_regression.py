import sys

sys.path.insert(0, ".")
import numpy as np
from quant.bench import profile_quantization
from quant.diagnose import diagnose_noop_layers
from quant.recover import recover_fp32_layer, recover_state_dict


def test_recover_fp32_layer_exact():
    packed = np.array([[0x12, 0x34]], dtype=np.uint8)
    scale = np.array([[1.0]], dtype=np.float32)
    zp = np.array([[0.0]], dtype=np.float32)
    res = recover_fp32_layer(packed, scale, zp, (1, 4), 4)
    expected = np.array([[2.0, 1.0, 4.0, 3.0]], dtype=np.float32)
    assert np.allclose(res, expected)


def test_recover_state_dict_structure():
    packed = np.array([[0x00]], dtype=np.uint8)
    scale = np.array([[1.0]], dtype=np.float32)
    sd = {
        "layer1.weight_packed": packed,
        "layer1.weight_scale": scale,
        "layer1.shape": (1, 2),
        "layer1.group_size": 2,
    }
    rec = recover_state_dict(sd)
    assert "layer1.weight" in rec
    assert rec["layer1.weight"].shape == (1, 2)


def test_diagnose_noop_layers():
    b = {"l1": {"dtype": "float32"}}
    a = {"l1": {"dtype": "float32"}}
    noops = diagnose_noop_layers(b, a)
    assert "l1" in noops
