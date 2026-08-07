import numpy as np


def vector_add_kernel_step(
    x: np.ndarray,
    y: np.ndarray,
    out: np.ndarray,
    pid: int,
    block_size: int,
    mask_boundary: bool = True,
) -> None:
    raise NotImplementedError


def run_vector_add(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    grid_type: str = "correct",
) -> tuple[np.ndarray, int]:
    raise NotImplementedError
