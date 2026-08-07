import numpy as np
from reference.batching.profiler import measure_latency_curve
from reference.batching.window import find_optimal_window
from reference.batching.queues import TieredQueueManager
from reference.batching.controller import DynamicBatchController

def get_oracle_curve(sizes):
    return measure_latency_curve(sizes)

def get_oracle_window(curve, slo):
    return find_optimal_window(curve, slo)
