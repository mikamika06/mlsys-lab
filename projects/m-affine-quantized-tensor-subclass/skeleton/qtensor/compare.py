import numpy as np
from qtensor.configs import map_target_to_config
from qtensor.subclass import quantize_affine


def get_rel_err(weight: np.ndarray, target: str) -> float:
    """
    Quantize the weight according to the target's config, dequantize it, 
    and return the relative Frobenius norm error: ||dequantized - weight|| / ||weight||
    """
    raise NotImplementedError
