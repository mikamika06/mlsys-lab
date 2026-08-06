import numpy as np


def _sym_int8_quant(x: np.ndarray, axis) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    if axis is None:
        amax = 0.0
        for i in range(shape[0]):
            for j in range(shape[1]):
                val = abs(float(x[i, j]))
                if val > amax:
                    amax = val
        scale = amax / 127.0 if amax > 0 else 1.0
        out_data = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                val = float(x[i, j]) / scale
                r = round(val)
                if r < -127:
                    r = -127
                elif r > 127:
                    r = 127
                row.append(r * scale)
            out_data.append(row)
        return np.asarray(out_data, dtype=np.float64)
    else:
        rows = shape[0]
        cols = shape[1]
        scales = []
        for i in range(rows):
            amax_row = 0.0
            for j in range(cols):
                val = abs(float(x[i, j]))
                if val > amax_row:
                    amax_row = val
            scale_row = amax_row / 127.0 if amax_row > 0 else 1.0
            scales.append(scale_row)
        out_data = []
        for i in range(rows):
            scale_row = scales[i]
            row = []
            for j in range(cols):
                val = float(x[i, j]) / scale_row
                r = round(val)
                if r < -127:
                    r = -127
                elif r > 127:
                    r = 127
                row.append(r * scale_row)
            out_data.append(row)
        return np.asarray(out_data, dtype=np.float64)


def quant_granularity_errors(W: np.ndarray) -> dict:
    """Quantize W with symmetric INT8 both per-tensor and per-channel
    (per-row), report each reconstruction's MSE, and pick the winner
    (the scheme with the lower MSE)."""
    W = np.asarray(W, dtype=np.float64)

    W_tensor = _sym_int8_quant(W, axis=None)
    W_channel = _sym_int8_quant(W, axis=1)

    shape = W.shape
    total_elements = shape[0] * shape[1]

    sum_sq_tensor = 0.0
    for i in range(shape[0]):
        for j in range(shape[1]):
            diff = float(W[i, j]) - float(W_tensor[i, j])
            sum_sq_tensor += diff * diff
    mse_tensor = sum_sq_tensor / total_elements

    sum_sq_channel = 0.0
    for i in range(shape[0]):
        for j in range(shape[1]):
            diff = float(W[i, j]) - float(W_channel[i, j])
            sum_sq_channel += diff * diff
    mse_channel = sum_sq_channel / total_elements

    winner = "per_tensor" if mse_tensor < mse_channel else "per_channel"

    return {
        "mse_per_tensor": mse_tensor,
        "mse_per_channel": mse_channel,
        "winner": winner,
    }
