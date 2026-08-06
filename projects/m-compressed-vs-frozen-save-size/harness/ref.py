import os
import json
import gzip
import numpy as np

CONFIGS = [
    {
        "model_id": "model_small",
        "weights": {"layer.0.weight": [16, 16], "layer.1.weight": [16, 16]}
    },
    {
        "model_id": "model_medium",
        "weights": {"layer.0.weight": [32, 32], "layer.1.weight": [32, 32], "layer.2.weight": [32, 32]}
    },
    {
        "model_id": "model_large",
        "weights": {"layer.0.weight": [64, 64]}
    }
]

def generate_fixtures(tmpdir):
    paths = []
    formats = ["raw", "compressed", "mmap"]
    for i, cfg in enumerate(CONFIGS):
        path = os.path.join(tmpdir, f"model_{i}.bin")
        data = json.dumps(cfg).encode("utf-8")
        if i == 1:
            data = gzip.compress(data)
        with open(path, "wb") as f:
            f.write(data)
        paths.append(path)
    return formats, paths
