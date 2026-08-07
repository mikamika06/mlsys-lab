import sys

sys.path.insert(0, ".")
from gguf_spec.diff import struct_diff

def test_diff_version_mismatch():
    a = {"version": 3}
    b = {"version": 2}
    d = struct_diff(a, b)
    assert len(d) == 1

def test_diff_identical():
    a = {"version": 3}
    b = {"version": 3}
    d = struct_diff(a, b)
    assert len(d) == 0
