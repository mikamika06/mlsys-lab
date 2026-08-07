import sys
import numpy as np

sys.path.insert(0, ".")
from quantlib.scheme import parse_quantization_config, compute_packed_shape
from quantlib.packer import pack_quantized_tensor


def test_parse_canonical_scheme():
    cfg = {
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "group_size": 128,
                    "symmetric": True,
                    "strategy": "group"
                }
            }
        }
    }
    parsed = parse_quantization_config(cfg)
    assert parsed["num_bits"] == 4
    assert parsed["group_size"] == 128
    assert parsed["symmetric"] is True
    assert parsed["strategy"] == "group"
    assert parsed["format"] == "pack-quantized"


def test_packed_shape_calculation():
    shape = (128, 512)
    packed_shape = compute_packed_shape(shape, num_bits=4, group_size=128, axis=-1)
    assert packed_shape == (128, 64)

    shape_unaligned = (128, 300)
    packed_unaligned = compute_packed_shape(shape_unaligned, num_bits=4, group_size=128, axis=-1)
    assert packed_unaligned == (128, 48)
