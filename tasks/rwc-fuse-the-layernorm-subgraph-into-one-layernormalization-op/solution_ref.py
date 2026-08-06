import math
import numpy as np


def _detect_span(nodes):
    expected = [
        "ReduceMean",
        "Sub",
        "Pow",
        "ReduceMean",
        "Add",
        "Sqrt",
        "Div",
        "Mul",
        "Add",
    ]
    if [n["op"] for n in nodes] == expected:
        return [n["name"] for n in nodes]
    return []


def fuse_layernorm_subgraph(nodes, inputs):
    x = np.asarray(inputs["x"], dtype=np.float64)
    gamma = np.asarray(inputs["gamma"], dtype=np.float64)
    beta = np.asarray(inputs["beta"], dtype=np.float64)
    eps = float(inputs.get("epsilon", 1e-5))

    shape = x.shape
    output = np.empty(shape, dtype=np.float64)

    if x.ndim == 1:
        n_features = shape[0]
        acc = 0.0
        for j in range(n_features):
            acc += x[j]
        mean_val = acc / n_features

        acc_var = 0.0
        for j in range(n_features):
            diff = x[j] - mean_val
            acc_var += diff * diff
        var_val = acc_var / n_features

        sqrt_val = math.sqrt(var_val + eps)

        for j in range(n_features):
            output[j] = gamma[j] * (x[j] - mean_val) / sqrt_val + beta[j]
    else:
        *batch_dims, n_features = shape
        flat_x = x.reshape(-1, n_features)
        flat_out = output.reshape(-1, n_features)
        n_batches = flat_x.shape[0]

        for i in range(n_batches):
            acc = 0.0
            for j in range(n_features):
                acc += flat_x[i, j]
            mean_val = acc / n_features

            acc_var = 0.0
            for j in range(n_features):
                diff = flat_x[i, j] - mean_val
                acc_var += diff * diff
            var_val = acc_var / n_features

            sqrt_val = math.sqrt(var_val + eps)

            for j in range(n_features):
                flat_out[i, j] = gamma[j] * (flat_x[i, j] - mean_val) / sqrt_val + beta[j]

    return {
        "fused_span": _detect_span(nodes),
        "output": output,
    }
