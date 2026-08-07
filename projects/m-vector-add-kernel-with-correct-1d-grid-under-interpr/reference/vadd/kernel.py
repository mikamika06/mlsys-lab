import numpy as np
from vadd.grid import get_grid_num_programs, get_underlaunched_num_programs


def vector_add_kernel_step(
    x: np.ndarray,
    y: np.ndarray,
    out: np.ndarray,
    pid: int,
    block_size: int,
    mask_boundary: bool = True,
) -> None:
    offsets = pid * block_size + np.arange(block_size)
    if mask_boundary:
        mask = offsets < len(x)
        valid = offsets[mask]
        out[valid] = x[valid] + y[valid]
    else:
        out[offsets] = x[offsets] + y[offsets]


def run_vector_add(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    grid_type: str = "correct",
) -> tuple[np.ndarray, int]:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float32)
    if grid_type == "correct":
        num_programs = get_grid_num_programs(n, block_size)
    elif grid_type == "underlaunched":
        num_programs = get_underlaunched_num_programs(n, block_size)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    for pid in range(num_programs):
        vector_add_kernel_step(x, y, out, pid, block_size, mask_boundary=True)

    computed_elements = min(n, num_programs * block_size)
    dropped_count = n - computed_elements
    return out, dropped_count
