def compute_shm_bytes(
    block_m: int, block_n: int, block_k: int, num_stages: int, dtype: str
) -> int:
    sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}
    sz = sizes.get(dtype, 2)
    return (block_m * block_k + block_k * block_n) * sz * num_stages
