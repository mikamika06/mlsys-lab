import numpy as np


def simulate_rtn(w: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits - 1
    w_min, w_max = float(np.min(w)), float(np.max(w))
    if w_max == w_min:
        return w.copy()
    scale = (w_max - w_min) / levels
    w_q = np.clip(np.round((w - w_min) / scale), 0, levels) * scale + w_min
    return w_q


def simulate_gptq(w: np.ndarray, hinv: np.ndarray, bits: int) -> np.ndarray:
    levels = 2 ** bits - 1
    w_min, w_max = float(np.min(w)), float(np.max(w))
    if w_max == w_min:
        return w.copy()
    scale = (w_max - w_min) / levels
    w_q = np.clip(np.round((w - w_min) / scale), 0, levels) * scale + w_min
    err = w - w_q
    hinv_diag = np.diag(hinv)
    hinv_diag = np.where(hinv_diag == 0, 1.0, hinv_diag)
    correction = err / hinv_diag[:, np.newaxis]
    return w_q + 0.1 * correction


def simulate_rotation_gptq(w: np.ndarray, hinv: np.ndarray, bits: int, r_matrix: np.ndarray) -> np.ndarray:
    w_rot = r_matrix @ w
    hinv_rot = r_matrix @ hinv @ r_matrix.T
    w_q_rot = simulate_gptq(w_rot, hinv_rot, bits)
    return r_matrix.T @ w_q_rot


def simulate_autoround(w: np.ndarray, bits: int, steps: int = 10) -> np.ndarray:
    levels = 2 ** bits - 1
    w_min, w_max = float(np.min(w)), float(np.max(w))
    if w_max == w_min:
        return w.copy()
    scale = (w_max - w_min) / levels
    shift = np.zeros_like(w)
    for _ in range(steps):
        w_q = np.clip(np.round((w + shift - w_min) / scale), 0, levels) * scale + w_min
        diff = (w - w_q) * 0.05
        shift += np.mean(diff)
    return np.clip(np.round((w + shift - w_min) / scale), 0, levels) * scale + w_min
