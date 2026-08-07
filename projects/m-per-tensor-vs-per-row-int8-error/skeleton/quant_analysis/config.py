import numpy as np


def pick_torchao_config(w: np.ndarray, max_mse: float) -> str:
    raise NotImplementedError


def select_model_configs(weights: dict[str, np.ndarray], max_mse: float) -> dict[str, str]:
    raise NotImplementedError
