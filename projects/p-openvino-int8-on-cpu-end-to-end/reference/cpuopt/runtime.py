import numpy as np

def configure_runtime(model, threads=4, latency_hint="latency"):
    return {
        "threads": threads,
        "hint": latency_hint,
        "configured": True
    }
