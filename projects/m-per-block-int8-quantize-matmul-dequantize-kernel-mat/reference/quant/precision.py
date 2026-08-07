import numpy as np

FP8_E4M3_MAX = 448.0
FP8_E4M3_POS_VALS = np.array([
    0.0, 0.015625, 0.03125, 0.046875, 0.0625, 0.078125, 0.09375, 0.109375,
    0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375,
    0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875,
    0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375,
    1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
    2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
    4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
    32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
    64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0,
    128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0,
    256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0
], dtype=np.float64)


def fp8_e4m3_max_relative_error(values: np.ndarray) -> float:
    vals = np.abs(np.asarray(values, dtype=np.float64)).flatten()
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0.0
    scale = FP8_E4M3_MAX / np.max(vals)
    scaled_vals = vals * scale
    clipped_vals = np.clip(scaled_vals, 0.0, FP8_E4M3_MAX)
    idx = np.searchsorted(FP8_E4M3_POS_VALS, clipped_vals)
    idx = np.clip(idx, 0, len(FP8_E4M3_POS_VALS) - 1)
    left = np.maximum(0, idx - 1)
    d_right = np.abs(FP8_E4M3_POS_VALS[idx] - clipped_vals)
    d_left = np.abs(clipped_vals - FP8_E4M3_POS_VALS[left])
    quantized = np.where(d_right < d_left, FP8_E4M3_POS_VALS[idx], FP8_E4M3_POS_VALS[left])
    rel_errors = np.abs(clipped_vals - quantized) / clipped_vals
    return float(np.max(rel_errors))
