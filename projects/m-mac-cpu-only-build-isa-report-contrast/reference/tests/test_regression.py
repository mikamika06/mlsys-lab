import sys
sys.path.insert(0, ".")
from isareport.parser import parse_build_log
from isareport.contrast import contrast_isa
from isareport.analyzer import analyze_tier

def test_parse_validates_native_flag():
    log = "CMAKE_ARGS='-DGGML_NATIVE=ON'\nDetected Neon: True\n"
    res = parse_build_log(log)
    assert res.get("native_flag") is True
    assert res.get("neon") is True

def test_contrast_identifies_differences():
    native = {"neon": True, "fp16": False}
    manual = {"neon": True, "fp16": True}
    diffs = contrast_isa(native, manual)
    assert "fp16" in diffs
    assert diffs["fp16"]["native"] is False
    assert diffs["fp16"]["manual"] is True

def test_tier_assignment():
    features = {"neon": True, "fp16": True}
    assert analyze_tier(features) == "T1"
    features_low = {"neon": True, "fp16": False}
    assert analyze_tier(features_low) == "T0"
