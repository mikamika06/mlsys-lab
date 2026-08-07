import sys
sys.path.insert(0, ".")
from coldstart.trace import parse_trace
from coldstart.tax import compute_cold_start_tax
from coldstart.scale import fraction_exposed


def test_parse_trace_sorting():
    raw = [{"arrival": 4.0}, {"arrival": 1.0}]
    res = parse_trace(raw)
    assert res[0]["arrival"] == 1.0


def test_tax_calculation():
    trace = [{"arrival": 0.0}, {"arrival": 10.0}]
    tax = compute_cold_start_tax(trace, 5.0, 1.0, 0.0)
    assert tax == 2.0


def test_fraction_exposed_bounds():
    trace = [{"arrival": float(i)} for i in range(5)]
    frac = fraction_exposed(trace, 10.0)
    assert 0.0 <= frac <= 1.0
    assert frac == 0.2
