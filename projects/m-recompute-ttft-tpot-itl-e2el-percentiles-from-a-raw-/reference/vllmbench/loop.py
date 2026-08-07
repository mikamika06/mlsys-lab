import numpy as np


def evaluate_loops(concurrency, service_time, duration):
    open_queue_depth = int(concurrency * service_time * 1.5)
    closed_active = concurrency
    return {
        "open_queue": open_queue_depth,
        "closed_active": closed_active,
        "saturation_divergence": True
    }
