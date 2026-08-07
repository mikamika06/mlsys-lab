import numpy as np

TEST_SIZES = [
    (1024, 128),
    (1000, 128),
    (1, 64),
    (5000, 256),
    (12345, 1024),
    (8000, 1024),
]


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_grid_num_programs(n: int, block_size: int) -> int:
    return cdiv(n, block_size)


def calculate_launch_waste(n: int, block_size: int) -> int:
    return get_grid_num_programs(n, block_size) * block_size - n


def get_underlaunched_num_programs(n: int, block_size: int) -> int:
    return n // block_size


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


def generate_input_pair(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    x = rng.randn(n).astype(np.float32)
    y = rng.randn(n).astype(np.float32)
    return x, y
