import sys
import os
sys.path.insert(0, ".")
from policy.quant import assign_kv_dtypes

CONFIG = [
    {"index": 0, "kind": "full"},
    {"index": 1, "kind": "sliding", "window": 1024},
    {"index": 2, "kind": "sliding", "window": 1024},
    {"index": 3, "kind": "full"},
]

def test_sliding_window_is_float16():
    dtypes = assign_kv_dtypes(CONFIG)
    for l, d in zip(CONFIG, dtypes):
        if l["kind"] == "sliding":
            assert d == "float16", f"layer {l['index']} is sliding but got {d}"

def test_full_is_float8():
    dtypes = assign_kv_dtypes(CONFIG)
    for l, d in zip(CONFIG, dtypes):
        if l["kind"] == "full":
            assert d == "float8", f"layer {l['index']} is full but got {d}"
