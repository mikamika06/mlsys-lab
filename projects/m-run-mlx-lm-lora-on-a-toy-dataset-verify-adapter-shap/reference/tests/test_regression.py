import sys

sys.path.insert(0, ".")
from loratools.adapter import verify_adapter_shape
from loratools.memory import measure_peak_rss
from loratools.params import compute_adapter_parameters


def test_adapter_shape_dimensions():
    shapes = verify_adapter_shape(256, 128, 16)
    assert shapes[0] == (128, 16)
    assert shapes[1] == (16, 256)


def test_memory_profile_relation():
    full_m = measure_peak_rss(100.0, False)
    qlora_m = measure_peak_rss(100.0, True)
    assert qlora_m < full_m


def test_parameter_count_computation():
    count = compute_adapter_parameters(["q_proj", "v_proj"], 8, 64, 64)
    assert count == 2 * (8 * (64 + 64))
