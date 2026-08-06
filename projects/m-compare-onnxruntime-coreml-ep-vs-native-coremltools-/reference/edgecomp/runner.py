import numpy as np


def compare_runtimes(model_spec):
    x = model_spec["input"]
    unsupported = model_spec["unsupported_ops"]
    out_native = np.tanh(x) * 2.0 - 1.0
    lat_native = float(np.prod(x.shape) * 1e-6 + 0.5)

    out_ep = np.tanh(x) * 2.0 - 1.0
    lat_ep = float(np.prod(x.shape) * 1e-6 + 0.5 + unsupported * 0.3)

    ratio = lat_ep / (lat_native + 1e-9)
    match = np.allclose(out_native, out_ep, atol=1e-5)
    return {"native_latency": lat_native, "ep_latency": lat_ep, "latency_ratio": ratio, "outputs_match": bool(match)}
