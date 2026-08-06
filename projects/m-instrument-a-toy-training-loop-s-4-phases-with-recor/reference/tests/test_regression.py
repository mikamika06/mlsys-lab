import sys

sys.path.insert(0, ".")
from profiler.trace import detect_unbalanced_ranges


def test_valid_nesting_passes():
    events = [
        {"type": "push", "name": "forward"},
        {"type": "push", "name": "layer1"},
        {"type": "pop", "name": "layer1"},
        {"type": "pop", "name": "forward"},
    ]
    res = detect_unbalanced_ranges(events)
    assert res["balanced"] is True
    assert res["error_index"] == -1


def test_mismatched_pop_detected():
    events = [
        {"type": "push", "name": "forward"},
        {"type": "pop", "name": "backward"},
    ]
    res = detect_unbalanced_ranges(events)
    assert res["balanced"] is False
    assert res["error_index"] == 1


def test_unclosed_push_detected():
    events = [
        {"type": "push", "name": "forward"},
        {"type": "push", "name": "loss"},
        {"type": "pop", "name": "loss"},
    ]
    res = detect_unbalanced_ranges(events)
    assert res["balanced"] is False
    assert res["error_index"] == 0
