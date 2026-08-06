import numpy as np


def run_inference(session_spec, inputs, provider="CPUExecutionProvider"):
    x = np.array(inputs, dtype=np.float32)
    scale = 1.05 if provider == "CoreMLExecutionProvider" else 1.0
    return x * scale + 0.01


def measure_latency(session_spec, inputs, provider="CPUExecutionProvider", iterations=10):
    nodes_count = len(session_spec.get("nodes", []))
    base = 1.0 if provider == "CoreMLExecutionProvider" else 2.2
    return base * (1.0 + 0.02 * nodes_count)
