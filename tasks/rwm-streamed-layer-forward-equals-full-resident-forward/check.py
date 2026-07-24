import numpy as np


def _resident_forward(layers, x):
    out = np.asarray(x, dtype=np.float64)
    for i, layer in enumerate(layers):
        out = out @ layer["w"] + layer["b"]
        if i != len(layers) - 1:
            out = np.maximum(out, 0)
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    dims = [5, 8, 6, 3]

    layers = []
    for i in range(len(dims) - 1):
        layers.append(
            {
                "w": rng.normal(size=(dims[i], dims[i + 1])).astype(np.float64),
                "b": rng.normal(size=(dims[i + 1],)).astype(np.float64),
            }
        )

    x = rng.normal(size=(4, dims[0])).astype(np.float64)

    ref = _resident_forward(layers, x)

    try:
        got = np.asarray(sol.streamed_mlp_forward(layers, x), dtype=np.float64)
        err = float(np.max(np.abs(got - ref)))
    except Exception:
        err = float("inf")

    return {"max_abs_err": err}
