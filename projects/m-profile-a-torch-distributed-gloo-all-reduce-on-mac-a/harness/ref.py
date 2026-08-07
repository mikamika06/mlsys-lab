import random

def generate_traces():
    random.seed(42)
    traces = []
    for i in range(5):
        t_type = "dp" if i % 2 == 0 else "tp"
        events = []
        self_time = 10.0 + i * 2.5
        events.append({
            "name": "gloo:all_reduce",
            "dur": self_time * 1000,
            "self_time": self_time,
            "cat": "operator",
            "args": {"comm_pattern": t_type}
        })
        traces.append({"events": events, "type": t_type, "expected_self_time": self_time})
    return traces

TRACES = generate_traces()
