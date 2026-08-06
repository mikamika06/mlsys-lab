def write_kv_cache(cache_k: list[list[float]], cache_v: list[list[float]], new_k: list[float], new_v: list[float], position: int) -> tuple[list[list[float]], list[list[float]]]:
    updated_k = [row[:] for row in cache_k]
    updated_v = [row[:] for row in cache_v]
    updated_k[position] = list(new_k)
    updated_v[position] = list(new_v)
    return updated_k, updated_v
