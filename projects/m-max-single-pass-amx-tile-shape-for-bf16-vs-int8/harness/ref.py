def max_tile_shape(dtype):
    if dtype == "bf16":
        return (16, 32)
    elif dtype == "int8":
        return (16, 64)
    raise ValueError(f"unknown dtype {dtype}")


def tmul_vs_avx512_ratio(dtype):
    if dtype == "bf16":
        return 4.0
    elif dtype == "int8":
        return 4.0
    raise ValueError(f"unknown dtype {dtype}")


def classify_tileability(M, N, K, dtype):
    rows, cols = max_tile_shape(dtype)
    single_pass_m = (M <= rows)
    single_pass_n = (N <= cols)
    single_pass_k = True
    return {
        "single_pass": bool(single_pass_m and single_pass_n and single_pass_k),
        "max_rows": rows,
        "max_cols": cols
    }


DATATYPES = ["bf16", "int8"]
