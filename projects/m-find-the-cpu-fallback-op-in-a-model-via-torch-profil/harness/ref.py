import torch


def generate_mock_trace_events():
    return [
        {"name": "aten::native_dropout", "dur": 10, "cat": "cpu_op"},
        {"name": "aten::nonzero", "dur": 5000, "cat": "cpu_op"},
        {"name": "aten::to", "dur": 1200, "cat": "Memcpy"},
        {"name": "aten::add", "dur": 5, "cat": "gpu_op"},
    ]


def identify_fallbacks(trace_events):
    fallbacks = []
    for ev in trace_events:
        if ev.get("cat") == "Memcpy" or (ev.get("cat") == "cpu_op" and "nonzero" in ev.get("name", "")):
            fallbacks.append(ev["name"])
    return sorted(list(set(fallbacks)))


def compute_latency_ratio(fallback_lat, native_lat):
    if native_lat == 0:
        return float("inf")
    return float(fallback_lat) / float(native_lat)
