import numpy as np


def cast_tensor_to_outtype(array: np.ndarray, outtype: str) -> np.ndarray:
    raise NotImplementedError


def compute_representation_error(state_dict: dict, outtype: str) -> dict:
    raise NotImplementedError
