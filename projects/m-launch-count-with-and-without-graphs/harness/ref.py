import numpy as np

TEST_CONFIGS = [
    {"op_sequence_len": 50, "steps": 100, "use_graph": False},
    {"op_sequence_len": 50, "steps": 100, "use_graph": True},
    {"op_sequence_len": 128, "steps": 10, "use_graph": False},
    {"op_sequence_len": 128, "steps": 10, "use_graph": True},
    {"op_sequence_len": 0, "steps": 20, "use_graph": True},
]

PREDICT_CONFIGS = [
    {"num_ops": 10, "kernel_gpu_time_us": 2.0, "host_launch_overhead_us": 5.0, "graph_launch_overhead_us": 2.0},
    {"num_ops": 100, "kernel_gpu_time_us": 0.5, "host_launch_overhead_us": 4.0, "graph_launch_overhead_us": 1.5},
    {"num_ops": 1, "kernel_gpu_time_us": 100.0, "host_launch_overhead_us": 5.0, "graph_launch_overhead_us": 2.0},
    {"num_ops": 0, "kernel_gpu_time_us": 10.0, "host_launch_overhead_us": 5.0, "graph_launch_overhead_us": 2.0},
]


def count_launches(op_sequence_len, steps, use_graph):
    if steps <= 0 or op_sequence_len < 0:
        return {"total_dispatches": 0, "graph_launches": 0, "individual_launches": 0}

    if use_graph:
        return {
            "total_dispatches": steps,
            "graph_launches": steps,
            "individual_launches": 0,
        }

    total = op_sequence_len * steps
    return {
        "total_dispatches": total,
        "graph_launches": 0,
        "individual_launches": total,
    }


def predict_speedup(num_ops, kernel_gpu_time_us, host_launch_overhead_us=5.0, graph_launch_overhead_us=2.0):
    if num_ops <= 0:
        return 1.0

    standard_time = num_ops * (kernel_gpu_time_us + host_launch_overhead_us)
    graph_time = (num_ops * kernel_gpu_time_us) + graph_launch_overhead_us

    if graph_time <= 0:
        return 1.0

    return standard_time / graph_time
