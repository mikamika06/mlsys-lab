import numpy as np


def classify_feasibility(config: dict, device_gb: float) -> bool:
    f32_bytes = np.dtype(np.float32).itemsize

    params = int(config["params"])
    batch = int(config["batch"])
    seq = int(config["seq"])
    hidden = int(config["hidden"])
    layers = int(config["layers"])

    total_bytes = (
        params * f32_bytes
        + params * f32_bytes
        + 2 * params * f32_bytes
        + batch * seq * hidden * layers * f32_bytes
    )

    return total_bytes <= float(device_gb) * 10**9
