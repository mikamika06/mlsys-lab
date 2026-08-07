import random

REQUESTS = [
    {"id": i, "model": f"model_{i%3}", "ctx": 512 * ((i%4)+1), "temp": 0.7}
    for i in range(12)
]

CONFIG_CHANGES = [
    {"parameter": "model_path", "old": "a.bin", "new": "b.bin"},
    {"parameter": "temperature", "old": 0.2, "new": 0.7},
    {"parameter": "gpu_layers", "old": 20, "new": 33},
    {"parameter": "n_predict", "old": 128, "new": 256},
    {"parameter": "tensor_split", "old": [0.5, 0.5], "new": [0.3, 0.7]},
    {"parameter": "seed", "old": 42, "new": 100},
    {"parameter": "rope_freq_base", "old": 10000.0, "new": 50000.0},
    {"parameter": "batch_size", "old": 512, "new": 1024},
]
