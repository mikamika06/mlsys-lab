import numpy as np


def get_test_cases():
    """Get reference test cases."""
    rng = np.random.RandomState(42)
    tensors = [
        rng.randn(64).astype(np.float32) * 50.0,
        rng.randn(64).astype(np.float32) * 200.0
    ]
    histories = [
        [rng.randn(32).astype(np.float32) * 10.0 for _ in range(5)],
        [rng.randn(32).astype(np.float32) * (1.0 if i < 4 else 100.0) for i in range(5)]
    ]
    scales = [1.5, 0.4, 2.0]
    return tensors, histories, scales


def compute_mse_optimal_scale(x, max_val=448.0, num_steps=30):
    """Reference optimal scale."""
    x_arr = np.asarray(x, dtype=np.float32)
    abs_x = np.abs(x_arr)
    max_abs = np.max(abs_x)
    if max_abs == 0:
        return 1.0
    base_scale = max_abs / max_val
    if base_scale == 0:
        return 1.0
    scales = base_scale * np.linspace(0.5, 1.5, num_steps)
    best_scale = scales[0]
    best_mse = float("inf")
    for s in scales:
        clipped = np.clip(x_arr, -s * max_val, s * max_val)
        mse = np.mean((x_arr - clipped) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = s
    return float(best_scale)


def decide_scaling_mode(activation_history, threshold=3.0):
    """Reference scaling mode decision."""
    history = [np.asarray(a, dtype=np.float32) for a in activation_history]
    if not history:
        return "static"
    maxs = [np.max(np.abs(a)) for a in history]
    mean_max = np.mean(maxs)
    peak_max = np.max(maxs)
    if mean_max == 0:
        return "static"
    ratio = peak_max / mean_max
    if ratio > threshold:
        return "dynamic"
    return "static"


def detect_inverted_scale(scale, reference_scale):
    """Reference inverted scale detection."""
    s = float(scale)
    ref = float(reference_scale)
    if s <= 0 or ref <= 0:
        return False
    dist_direct = abs(s - ref)
    dist_inverse = abs(s - (1.0 / ref))
    return dist_inverse < dist_direct
