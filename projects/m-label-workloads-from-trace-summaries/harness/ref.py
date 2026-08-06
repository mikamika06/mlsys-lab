import numpy as np

TRACES = [
    [
        {"name": "aten::add", "dur_us": 12.0, "launch_delay_us": 25.0, "flops": 1000, "bytes": 4000},
        {"name": "aten::mm", "dur_us": 150.0, "launch_delay_us": 5.0, "flops": 1e9, "bytes": 1e6},
        {"name": "aten::empty", "dur_us": 2.0, "launch_delay_us": 3.0, "flops": 0, "bytes": 0},
    ],
    [
        {"name": "aten::relu", "dur_us": 8.0, "launch_delay_us": 30.0, "flops": 500, "bytes": 2000},
        {"name": "aten::bmm", "dur_us": 200.0, "launch_delay_us": 4.0, "flops": 2e9, "bytes": 2e6},
    ],
    [
        {"name": "aten::copy_", "dur_us": 5.0, "launch_delay_us": 20.0, "flops": 0, "bytes": 10000},
        {"name": "aten::linear", "dur_us": 300.0, "launch_delay_us": 2.0, "flops": 5e9, "bytes": 5e6},
    ]
]

HARDWARE_SPECS = {
    "peak_tflops": 312.0,
    "peak_bandwidth_gbps": 2000.0,
}

OP_MIXES = [
    [
        {"flops": 1e12, "bytes": 1e9},
        {"flops": 5e11, "bytes": 8e9},
    ],
    [
        {"flops": 1e8, "bytes": 1e8},
        {"flops": 2e8, "bytes": 4e8},
    ]
]

SYNC_EVENT_LOGS = [
    [
        {"event": "kernel_launch", "ts_us": 10.0, "stream": 1, "is_blocking": False},
        {"event": "tensor_item_call", "ts_us": 15.0, "stream": 1, "is_blocking": True},
        {"event": "cudaStreamSynchronize", "ts_us": 20.0, "stream": 1, "is_blocking": True},
    ],
    [
        {"event": "kernel_launch", "ts_us": 5.0, "stream": 1, "is_blocking": False},
        {"event": "host_to_device_non_pinned", "ts_us": 8.0, "stream": 1, "is_blocking": True},
    ]
]


def classify_op(event):
    if event["flops"] == 0 and event["bytes"] == 0:
        return "overhead"
    if event["launch_delay_us"] > event["dur_us"]:
        return "launch_bound"
    ai = event["flops"] / max(event["bytes"], 1)
    if ai > 100.0:
        return "compute_bound"
    return "memory_bound"


def label_workloads(trace):
    return [classify_op(ev) for ev in trace]


def analyze_roofline(op_mix, spec):
    total_flops = sum(op["flops"] for op in op_mix)
    total_bytes = sum(op["bytes"] for op in op_mix)
    ai = total_flops / max(total_bytes, 1.0)
    knee_point = (spec["peak_tflops"] * 1e12) / (spec["peak_bandwidth_gbps"] * 1e9)
    attained_tflops = min(spec["peak_tflops"], (ai * spec["peak_bandwidth_gbps"] * 1e9) / 1e12)
    bound = "compute_bound" if ai >= knee_point else "memory_bound"
    return {
        "intensity": ai,
        "knee_point": knee_point,
        "attained_tflops": attained_tflops,
        "bound": bound
    }


def find_hidden_syncs(event_log):
    return [i for i, ev in enumerate(event_log) if ev.get("is_blocking", False)]
