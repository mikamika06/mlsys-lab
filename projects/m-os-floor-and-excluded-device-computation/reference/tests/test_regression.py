import sys

sys.path.insert(0, ".")
from edgeexport.convert import convert_variant_manifest
from edgeexport.filtering import compute_os_floor, filter_eligible_devices
from edgeexport.selection import select_variant_set


def test_os_floor_computation():
    feat_map = {"fp16": (14, 0), "metal3": (16, 0)}
    var = {"min_os": (14, 0), "required_features": ["metal3"]}
    assert compute_os_floor(var, feat_map) == (16, 0)


def test_budget_is_strictly_respected():
    variants = [
        {"id": "v1", "download_bytes": 100, "utility": 10.0},
        {"id": "v2", "download_bytes": 200, "utility": 25.0},
        {"id": "v3", "download_bytes": 150, "utility": 20.0},
    ]
    budget = 250
    selected = select_variant_set(variants, budget)
    selected_vars = [v for v in variants if v["id"] in selected]
    total_size = sum(v["download_bytes"] for v in selected_vars)
    assert total_size <= budget, f"total size {total_size} exceeds budget {budget}"


def test_bit_reproducible_hash():
    m1 = {"b": [1.0000001], "a": 2.0}
    m2 = {"a": 2.0, "b": [1.0]}
    res1 = convert_variant_manifest(m1)
    res2 = convert_variant_manifest(m2)
    assert res1["digest"] == res2["digest"]
