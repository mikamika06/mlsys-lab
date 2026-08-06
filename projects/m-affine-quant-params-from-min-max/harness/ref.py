import numpy as np


def get_test_ranges():
    return [
        (-1.0, 1.0, 0, 255),
        (0.0, 10.0, 0, 255),
        (-5.0, 0.0, 0, 255),
        (2.0, 8.0, 0, 255),
        (-12.0, -3.0, 0, 255),
        (-2.5, 3.5, -128, 127),
    ]


def calc_affine_params_ref(val_min: float, val_max: float, qmin: int = 0, qmax: int = 255) -> tuple[float, int]:
    r_min = min(val_min, 0.0)
    r_max = max(val_max, 0.0)
    if r_min == r_max:
        return 1.0, qmin
    scale = (r_max - r_min) / float(qmax - qmin)
    initial_zp = qmin - (r_min / scale)
    zp = int(np.round(initial_zp))
    zp_clamped = max(qmin, min(qmax, zp))
    return float(scale), zp_clamped


def generate_calibration_dataset(seed: int = 42) -> list[dict[str, np.ndarray]]:
    rng = np.random.RandomState(seed)
    batches = []
    for _ in range(5):
        batch = {
            "conv1_out": rng.normal(loc=1.0, scale=2.0, size=(4, 8)).astype(np.float32),
            "relu1_out": np.maximum(0.0, rng.normal(loc=0.5, scale=1.5, size=(4, 8))).astype(np.float32),
            "linear_out": rng.uniform(low=-10.0, high=10.0, size=(4, 8)).astype(np.float32),
        }
        batches.append(batch)
    return batches
