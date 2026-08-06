import random


def generate_trace():
    random.seed(42)
    events = [
        {
            "name": "triton_fused_attention_kernel",
            "dur": 150000,
            "args": {"memory": 1024},
        },
        {
            "name": "triton_gemm_kernel",
            "dur": 300000,
            "args": {"memory": 2048},
        },
        {
            "name": "cross_entropy_loss",
            "dur": 50000,
            "args": {"memory": 8192},
        },
        {
            "name": "elementwise_add",
            "dur": 20000,
            "args": {"memory": 512},
        },
    ]
    return {"traceEvents": events}


def top_time_kernel(trace):
    totals = {}
    for ev in trace.get("traceEvents", []):
        name = ev.get("name", "")
        if "kernel" in name or "loss" in name or "add" in name:
            totals[name] = totals.get(name, 0) + ev.get("dur", 0)
    if not totals:
        return ""
    return max(totals, key=totals.get)


def cross_entropy_memory_share(trace):
    total_mem = 0
    ce_mem = 0
    for ev in trace.get("traceEvents", []):
        mem = ev.get("args", {}).get("memory", 0)
        total_mem += mem
        if "cross_entropy" in ev.get("name", ""):
            ce_mem += mem
    if total_mem == 0:
        return 0.0
    return float(ce_mem) / float(total_mem)


TRACES = [generate_trace()]
