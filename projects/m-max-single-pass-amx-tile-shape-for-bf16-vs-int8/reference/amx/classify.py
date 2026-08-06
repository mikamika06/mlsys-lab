def is_single_pass_tileable(m: int, n: int, k: int, dtype: str) -> bool:
    max_m, max_n = (16, 32) if dtype == "bf16" else (16, 64)
    return m <= max_m and n <= max_n and k <= 64
