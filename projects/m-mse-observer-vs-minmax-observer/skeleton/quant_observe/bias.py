import numpy as np
from quant_observe.observer import minmax_observer, mse_observer

def ignored_zp_bias(x: np.ndarray, args: dict, method: str) -> float:
    raise NotImplementedError
