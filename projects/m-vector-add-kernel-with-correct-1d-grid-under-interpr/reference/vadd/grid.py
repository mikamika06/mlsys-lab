def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_grid_num_programs(n: int, block_size: int) -> int:
    return cdiv(n, block_size)


def calculate_launch_waste(n: int, block_size: int) -> int:
    return get_grid_num_programs(n, block_size) * block_size - n


def get_underlaunched_num_programs(n: int, block_size: int) -> int:
    return n // block_size
