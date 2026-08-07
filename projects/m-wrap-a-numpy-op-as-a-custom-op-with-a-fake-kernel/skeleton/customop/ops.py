import torch
import numpy as np

def custom_op_func(x: torch.Tensor, scale: float) -> torch.Tensor:
    raise NotImplementedError

def find_bad_dim(shape_list, expected_shape) -> int:
    raise NotImplementedError
