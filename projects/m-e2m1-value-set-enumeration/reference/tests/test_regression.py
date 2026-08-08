import sys
import math

sys.path.insert(0, ".")
from microscale.e2m1 import decode_e2m1, enumerate_values


def test_zero_is_representable():
    v0 = decode_e2m1(0, bias=1, has_nan=False, has_inf=False)
    v8 = decode_e2m1(8, bias=1, has_nan=False, has_inf=False)
    assert v0 == 0.0, f"Expected exactly 0.0 for E=0 M=0, got {v0}"
    assert v8 == 0.0, f"Expected exactly 0.0 for E=0 M=0, got {v8}"


def test_subnormal_scale():
    v1 = decode_e2m1(1, bias=1, has_nan=False, has_inf=False)
    assert v1 == 0.5, f"Expected subnormal 0.5, got {v1}"


def test_symmetry():
    vals = enumerate_values(1, False, False)
    positives = [v for v in vals if v > 0]
    negatives = [v for v in vals if v < 0]
    for p in positives:
        assert -p in negatives, f"Value {-p} not found symmetrically for {p}"
