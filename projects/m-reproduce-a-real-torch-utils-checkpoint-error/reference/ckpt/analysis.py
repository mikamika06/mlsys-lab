import numpy as np

def measure_memory_time(num_layers, checkpoint_interval):
    base_mem = num_layers * 10.0
    if checkpoint_interval > 0:
        ckpt_mem = (num_layers / checkpoint_interval) * 10.0 + checkpoint_interval * 5.0
    else:
        ckpt_mem = base_mem
    time_cost = float(num_layers) + (float(num_layers) / float(checkpoint_interval) if checkpoint_interval > 0 else 0.0) * 0.2
    return {"memory": float(ckpt_mem), "time": float(time_cost)}

def optimal_interval(n_layers):
    return int(np.round(np.sqrt(n_layers)))
