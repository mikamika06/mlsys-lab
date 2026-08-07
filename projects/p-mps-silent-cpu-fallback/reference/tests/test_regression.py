import sys
sys.path.insert(0, ".")
from mps.engine import Engine


def test_no_fallbacks_in_standard_graph():
    eng = Engine()
    graph = [{"name": "add", "base_duration": 1.0}, {"name": "mul", "base_duration": 1.0}]
    trace = eng.run(graph)
    assert all(not step["fallback"] for step in trace)


def test_unimplemented_detection():
    eng = Engine()
    graph = [{"name": "unsupported_custom_op", "base_duration": 2.0}]
    unimpl = eng.list_unimplemented_ops(graph)
    assert "unsupported_custom_op" in unimpl
