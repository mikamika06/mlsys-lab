def max_tile_shape(dtype: str) -> tuple:
    if dtype == "bf16":
        return (16, 32)
    elif dtype == "int8":
        return (16, 64)
    raise ValueError(f"Unknown dtype: {dtype}")
