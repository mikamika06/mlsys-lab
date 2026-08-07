import sys
import numpy as np

sys.path.insert(0, ".")
from numprec.properties import get_format_properties, compute_relative_error
from numprec.autocast_rules import predict_autocast_action
from numprec.overflow import fp16_overflow_tail_probability


def test_fp16_properties_consistency():
    p = get_format_properties("fp16")
    assert p["exponent_bits"] == 5
    assert p["mantissa_bits"] == 10
    assert abs(p["max_val"] - 65504.0) < 1e-3
    assert abs(p["eps"] - 2.0**(-10)) < 1e-7


def test_bf16_properties_consistency():
    p = get_format_properties("bf16")
    assert p["exponent_bits"] == 8
    assert p["mantissa_bits"] == 7
    assert p["eps"] == 2.0**(-7)


def test_relative_error_exact_power_of_two():
    err = compute_relative_error(1.0, "fp16")
    assert err == 0.0


def test_autocast_action_classification():
    assert predict_autocast_action("matmul") == "cast"
    assert predict_autocast_action("softmax") == "keep_fp32"
    assert predict_autocast_action("add") == "promote"


def test_overflow_probability_limits():
    p_small = fp16_overflow_tail_probability(100.0)
    p_large = fp16_overflow_tail_probability(30000.0)
    assert p_small < 1e-12
    assert p_large > 0.01
