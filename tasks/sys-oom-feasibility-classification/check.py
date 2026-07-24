import numpy as np


def _oracle(config, device_gb):
    f32_bytes = np.dtype(np.float32).itemsize

    params = int(config["params"])
    batch = int(config["batch"])
    seq = int(config["seq"])
    hidden = int(config["hidden"])
    layers = int(config["layers"])

    param_bytes = params * f32_bytes
    grad_bytes = params * f32_bytes
    optimizer_bytes = 2 * params * f32_bytes
    activation_bytes = batch * seq * hidden * layers * f32_bytes

    total = param_bytes + grad_bytes + optimizer_bytes + activation_bytes
    capacity = float(device_gb) * 10**9
    return total <= capacity


def grade(sol, fx) -> dict:
    cases = [
        (
            {
                "params": 1_000_000,
                "batch": 2,
                "seq": 16,
                "hidden": 32,
                "layers": 4,
            },
            1.0,
        ),
        (
            {
                "params": 1_000_000_000,
                "batch": 8,
                "seq": 2048,
                "hidden": 4096,
                "layers": 32,
            },
            80.0,
        ),
        (
            {
                "params": 50_000_000,
                "batch": 16,
                "seq": 512,
                "hidden": 768,
                "layers": 12,
            },
            8.0,
        ),
        (
            {
                "params": 500_000_000,
                "batch": 4,
                "seq": 1024,
                "hidden": 1024,
                "layers": 24,
            },
            12.0,
        ),
        (
            {
                "params": 10_000_000,
                "batch": 1,
                "seq": 128,
                "hidden": 256,
                "layers": 6,
            },
            1.0,
        ),
    ]

    ok = 1.0
    for config, gb in cases:
        expected = _oracle(config, gb)
        try:
            got = bool(sol.classify_feasibility(dict(config), gb))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
