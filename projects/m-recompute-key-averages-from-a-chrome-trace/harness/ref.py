import random

def generate_traces():
    random.seed(42)
    events_a = [
        {"ph": "X", "cat": "gpu", "name": "aten::mm", "ts": 100, "dur": 200, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "gpu", "name": "aten::relu", "ts": 310, "dur": 50, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "gpu", "name": "aten::add", "ts": 370, "dur": 80, "pid": 1, "tid": 1},
    ]
    events_b = [
        {"ph": "X", "cat": "gpu", "name": "aten::mm", "ts": 100, "dur": 500, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "gpu", "name": "aten::relu", "ts": 610, "dur": 50, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "gpu", "name": "aten::add", "ts": 670, "dur": 80, "pid": 1, "tid": 1},
    ]
    return {"traceEvents": events_a}, {"traceEvents": events_b}
