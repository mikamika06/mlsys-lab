import numpy as np


def compute_reference_tasks(module_spec):
    tasks = []
    for name in module_spec:
        tasks.append({"task_name": f"f_{name}", "weight": 1.0})
    return tasks


def compute_reference_curve(trials_list, base_latency, min_latency):
    latencies = []
    for t in trials_list:
        lat = min_latency + (base_latency - min_latency) / (1.0 + 0.05 * t)
        latencies.append(float(lat))
    return latencies
