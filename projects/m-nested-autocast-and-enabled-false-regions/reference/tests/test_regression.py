import sys

sys.path.insert(0, ".")
from nested_cast.context import resolve_effective_states

CONFIG = {
    "device_type": "cuda",
    "children": [
        {"enabled": True, "dtype": "float16", "children": [
            {"enabled": False, "dtype": "float32", "children": []}
        ]}
    ]
}


def test_enabled_false_override():
    res = resolve_effective_states(CONFIG)
    inner_child = res["children"][0]["children"][0]
    assert inner_child["enabled"] is False, "enabled=False region failed to disable autocast"


def test_dtype_inheritance():
    res = resolve_effective_states(CONFIG)
    outer_child = res["children"][0]
    assert outer_child["dtype"] == "float16"
