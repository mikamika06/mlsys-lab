import numpy as np
from mlsys import scorers


def _oracle_quant(W):
    W = np.asarray(W, dtype=np.float64)

    tensor_scale = np.max(np.abs(W)) / 127.0
    tensor_q = np.clip(np.round(W / tensor_scale), -127, 127)
    tensor_out = tensor_q * tensor_scale

    channel_scale = np.max(np.abs(W), axis=1, keepdims=True) / 127.0
    channel_q = np.clip(np.round(W / channel_scale), -127, 127)
    channel_out = channel_q * channel_scale

    return tensor_out, channel_out


def grade(sol, fx) -> dict:
    cases = [
        np.array([
            [0.10, 0.20, 0.30, 0.40],
            [8.0, -7.0, 6.0, -5.0],
            [1.0, -1.5, 0.5, 2.0],
        ], dtype=np.float64),
        np.array([
            [0.01, 0.02, -0.03, 0.04],
            [20.0, -18.0, 15.0, 10.0],
            [2.0, 1.0, -1.0, -2.0],
            [0.5, -0.25, 0.75, 1.0],
        ], dtype=np.float64),
    ]

    tensor_errors = []
    candidate_channel_errors = []
    oracle_ok = 1.0

    for W in cases:
        try:
            tensor_out, channel_out = sol.compare_quant_errors(W.copy())
        except Exception:
            return {
                "channel_rel_err": 1.0,
                "improvement_margin": 0.0,
                "oracle_match": 0.0,
            }

        ref_tensor, ref_channel = _oracle_quant(W)

        if not np.allclose(tensor_out, ref_tensor, rtol=0, atol=1e-12):
            oracle_ok = 0.0
        if not np.allclose(channel_out, ref_channel, rtol=0, atol=1e-12):
            oracle_ok = 0.0

        tensor_errors.append(scorers.channel_rel_err(W, ref_tensor))
        candidate_channel_errors.append(scorers.channel_rel_err(W, channel_out))

    tensor_error = float(np.mean(tensor_errors))
    channel_error = float(np.mean(candidate_channel_errors))

    return {
        "channel_rel_err": channel_error,
        "improvement_margin": tensor_error - channel_error,
        "oracle_match": oracle_ok,
    }
