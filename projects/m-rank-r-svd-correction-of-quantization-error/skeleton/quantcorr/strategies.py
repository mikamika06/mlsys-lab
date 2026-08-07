import numpy as np


def simulate_rtn(w: np.ndarray, bits: int) -> np.ndarray:
    raise NotImplementedError


def simulate_gptq(w: np.ndarray, hinv: np.ndarray, bits: int) -> np.ndarray:
    raise NotImplementedError


def simulate_rotation_gptq(w: np.ndarray, hinv: np.ndarray, bits: int, r_matrix: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def simulate_autoround(w: np.ndarray, bits: int, steps: int = 10) -> np.ndarray:
    raise NotImplementedError
