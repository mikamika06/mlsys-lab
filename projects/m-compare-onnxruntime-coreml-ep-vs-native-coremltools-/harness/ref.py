import numpy as np


def generate_fixtures():
    rng = np.random.default_rng(42)
    models = []
    for i in range(5):
        shape = (1, 3, 32 + i * 16, 32 + i * 16)
        x = rng.standard_normal(shape).astype(np.float32)
        unsupported_count = i
        models.append({"id": i, "input": x, "unsupported_ops": unsupported_count})
    return models


def compute_native_execution(model):
    x = model["input"]
    out = np.tanh(x) * 2.0 - 1.0
    latency = float(np.prod(x.shape) * 1e-6 + 0.5)
    return {"output": out, "latency": latency}


def compute_coreml_ep_execution(model):
    x = model["input"]
    out = np.tanh(x) * 2.0 - 1.0
    base_latency = float(np.prod(x.shape) * 1e-6 + 0.5)
    fallback_penalty = float(model["unsupported_ops"] * 0.3)
    latency = base_latency + fallback_penalty
    return {"output": out, "latency": latency}


def compute_cpu_only_execution(model):
    x = model["input"]
    out = np.tanh(x) * 2.0 - 1.0
    base_latency = float(np.prod(x.shape) * 1e-6 + 2.0)
    return {"output": out, "latency": base_latency}
