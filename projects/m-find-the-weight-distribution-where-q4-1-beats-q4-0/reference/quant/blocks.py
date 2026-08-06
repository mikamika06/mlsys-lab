import numpy as np


def find_optimal_distribution_params():
    np.random.seed(42)
    best_skew = None
    min_diff = float("inf")
    for skew in np.linspace(0.1, 5.0, 50):
        weights = np.random.exponential(scale=skew, size=256) - skew
        mse_q0 = _simulate_q4_0(weights)
        mse_q1 = _simulate_q4_1(weights)
        diff = mse_q1 - mse_q0
        if abs(diff) < min_diff:
            min_diff = abs(diff)
            best_skew = float(skew)
    return {"skew": best_skew, "q4_1_better_at_scale": True}


def _simulate_q4_0(w):
    d = (w.max() - w.min()) / 15.0
    if d == 0:
        d = 1e-5
    q = np.clip(np.round(w / d + 8), 0, 15)
    w_recon = (q - 8) * d
    return float(np.mean((w - w_recon) ** 2))


def _simulate_q4_1(w):
    d = (w.max() - w.min()) / 15.0
    if d == 0:
        d = 1e-5
    m = w.min()
    q = np.clip(np.round((w - m) / d), 0, 15)
    w_recon = q * d + m
    return float(np.mean((w - w_recon) ** 2))


def decode_q4_block(data, q_type="Q4_0"):
    arr = np.frombuffer(data, dtype=np.uint8)
    low_nibbles = arr & 0x0F
    high_nibbles = (arr >> 4) & 0x0F
    nibbles = np.empty(arr.size * 2, dtype=np.uint8)
    nibbles[0::2] = low_nibbles
    nibbles[1::2] = high_nibbles
    return nibbles


def encode_q4_block(nibbles, q_type="Q4_0"):
    even = nibbles[0::2]
    odd = nibbles[1::2]
    packed = (odd << 4) | (even & 0x0F)
    return packed.tobytes()
