import sys

sys.path.insert(0, ".")
from quant.modes import analyze_op_compatibility, compare_int8_vs_int16x8

AUDIO_SR_MODEL = {
    "ops": [
        {"type": "conv1d"},
        {"type": "add"},
        {"type": "mul"}
    ],
    "weights": [
        {"name": "conv1", "count": 1024, "channels": 16}
    ]
}


def test_dynamic_range_vs_full_integer_op_requirements():
    dr = analyze_op_compatibility(AUDIO_SR_MODEL, "dynamic_range")
    i8 = analyze_op_compatibility(AUDIO_SR_MODEL, "int8")
    dr_add = next(o for o in dr if o["type"] == "add")
    i8_add = next(o for o in i8 if o["type"] == "add")
    assert dr_add["executes_float"] is True
    assert i8_add["executes_float"] is False
    assert i8_add["requires_calibration"] is True


def test_int16x8_scale_overflow():
    comp = compare_int8_vs_int16x8(AUDIO_SR_MODEL)
    assert comp["int16x8"]["weight_bytes"] == 2 * comp["int8"]["weight_bytes"]
    assert comp["size_ratio"] == 2.0
