import sys

sys.path.insert(0, ".")
from jaxpr_utils.analyzer import count_equations, list_unique_primitives, safe_trace_collector


def test_equation_counting():
    mock_jaxpr = {"eqns": [{"primitive": "add"}, {"primitive": "mul"}, {"primitive": "add"}]}
    assert count_equations(mock_jaxpr) == 3


def test_unique_primitives_order():
    mock_jaxpr = {"eqns": [{"primitive": "mul"}, {"primitive": "add"}, {"primitive": "mul"}]}
    assert list_unique_primitives(mock_jaxpr) == ["add", "mul"]


def test_closure_leak_detection():
    state = []
    def bad_fn(x):
        state.append(x)
        return x + 1

    out = safe_trace_collector(bad_fn, [1, 2])
    assert len(state) == 2
