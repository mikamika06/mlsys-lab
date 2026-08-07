import time
import numpy as np


def measure_throughput(logits, temperature):
    start = time.time()
    _ = np.argmax(logits, axis=-1)
    dur = time.time() - start
    return float(len(logits) / max(dur, 1e-6))
