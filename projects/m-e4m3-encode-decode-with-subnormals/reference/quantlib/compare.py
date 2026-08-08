import numpy as np
from quantlib.codec import encode_e4m3, decode_e4m3
from quantlib.scale import compute_scale


def compare_formats(x):
    x = np.asarray(x, dtype=np.float32)
    scale = compute_scale(x)
    scaled = x / scale
    encoded = encode_e4m3(scaled)
    decoded = decode_e4m3(encoded) * scale
    mse = float(np.mean((x - decoded) ** 2))
    assert mse >= 0.0
    return {"e4m3_mse": mse, "scale": scale}
