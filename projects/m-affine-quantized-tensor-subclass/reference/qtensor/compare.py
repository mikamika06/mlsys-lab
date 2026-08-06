import numpy as np
from qtensor.configs import map_target_to_config
from qtensor.subclass import quantize_affine


def get_rel_err(weight: np.ndarray, target: str) -> float:
    config = map_target_to_config(target)
    q_tensor = quantize_affine(weight, config["group_size"], config["asymmetric"])
    deq = q_tensor.dequantize()
    return float(np.linalg.norm(deq - weight) / np.linalg.norm(weight))
