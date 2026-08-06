from kvcalc.calc import bytes_per_token


def cache_ratio(config_mla: dict, config_gqa: dict, dtype_bytes: int = 2) -> float:
    """Calculate per-token cache ratio between MLA and GQA."""
    b_mla = bytes_per_token(config_mla, dtype_bytes)
    b_gqa = bytes_per_token(config_gqa, dtype_bytes)
    return float(b_mla) / float(b_gqa)
