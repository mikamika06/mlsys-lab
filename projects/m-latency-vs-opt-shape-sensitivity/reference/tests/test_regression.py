"""Regression tests for profile splitting and tensor classification."""

import sys
sys.path.insert(0, ".")

from trtopt.tensors import classify_tensors
from trtopt.profile import split_wide_profile, evaluate_profile_latency

def test_split_profiles_cover_range():
    wide = {"min": [1, 64, 512], "opt": [16, 64, 512], "max": [32, 64, 512]}
    sub_profiles = split_wide_profile(wide)
    assert len(sub_profiles) == 2
    p1, p2 = sub_profiles
    assert p1["min"][0] == wide["min"][0]
    assert p2["max"][0] == wide["max"][0]
    assert p1["max"][0] >= p2["min"][0]

def test_shape_tensors_identified():
    spec = {
        "inputs": [
            {"name": "input_ids", "is_shape_tensor": False, "role": "execution"},
            {"name": "shape_input", "is_shape_tensor": True, "role": "shape"}
        ]
    }
    res = classify_tensors(spec)
    assert "input_ids" in res["execution_tensors"]
    assert "shape_input" in res["shape_tensors"]

def test_profiles_not_merged():
    wide = {"min": [1, 64, 512], "opt": [16, 64, 512], "max": [32, 64, 512]}
    sub_profiles = split_wide_profile(wide)
    assert len(sub_profiles) == 2
    assert sub_profiles[0]["opt"][0] != sub_profiles[1]["opt"][0]
