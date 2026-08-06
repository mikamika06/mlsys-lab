from profiler.analyzer import hot_op_cover

def test_hot_op_cover_exact():
    events = [
        {"cat": "Session", "name": "model_run", "ts": 0, "dur": 100},
        {"cat": "Node", "name": "n1", "ts": 0, "dur": 40, "args": {"op_name": "A"}},
        {"cat": "Node", "name": "n2", "ts": 40, "dur": 40, "args": {"op_name": "B"}},
        {"cat": "Node", "name": "n3", "ts": 80, "dur": 20, "args": {"op_name": "C"}},
    ]
    assert hot_op_cover(events, 0.8) == {"A", "B"}
