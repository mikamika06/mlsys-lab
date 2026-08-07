import numpy as np

def estimate_workers(batch_time, worker_latency):
    return int(np.ceil(batch_time / worker_latency))
