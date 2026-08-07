import numpy as np

def sample_trace():
    return [
        {"name": "layer_0", "start": 0, "dur": 50, "type": "compute"},
        {"name": "all_reduce_0", "start": 50, "dur": 30, "type": "comm"},
        {"name": "layer_1", "start": 80, "dur": 50, "type": "compute"},
        {"name": "all_reduce_1", "start": 130, "dur": 30, "type": "comm"}
    ]

def sample_model_config():
    return {
        "hidden_size": 2048,
        "num_layers": 16,
        "dtype_bytes": 2
    }

def sample_tensors():
    return [{"id": i, "size": 1024 * 1024} for i in range(10)]
