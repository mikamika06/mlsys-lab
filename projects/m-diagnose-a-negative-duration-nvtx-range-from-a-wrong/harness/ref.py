def generate_scenario():
    events = [
        {"pid": 100, "tid": 1, "ts": 1000, "name": "Iteration", "ph": "B"},
        {"pid": 100, "tid": 1, "ts": 1100, "name": "Forward", "ph": "B"},
        {"pid": 100, "tid": 2, "ts": 1150, "name": "Forward", "ph": "E"},
        {"pid": 100, "tid": 1, "ts": 1200, "name": "Forward", "ph": "E"},
        {"pid": 100, "tid": 1, "ts": 1300, "name": "Iteration", "ph": "E"}
    ]
    phases = [
        {"name": "aten::matmul", "dur": 500, "safedur": 200, "self": 200},
        {"name": "aten::add", "dur": 100, "safedur": 100, "self": 100},
        {"name": "aten::gelu", "dur": 300, "safedur": 50, "self": 50},
        {"name": "aten::layer_norm", "dur": 400, "safedur": 150, "self": 150},
        {"name": "aten::copy_", "dur": 80, "safedur": 80, "self": 80},
        {"name": "optimizer_step", "dur": 1000, "safedur": 400, "self": 400}
    ]
    return events, phases
