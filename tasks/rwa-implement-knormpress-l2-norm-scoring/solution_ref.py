import numpy as np

def knormpress(data: dict[int, np.ndarray], budget: int) -> list[int]:
    norms = {k: np.linalg.norm(v) for k, v in data.items()}
    sorted_keys = sorted(norms, key=norms.get, reverse=True)
    top_k = sorted(sorted_keys[:budget])
    return top_k
