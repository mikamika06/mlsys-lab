import numpy as np

def verify_rslora_scaling(rank: int, alpha: float, method: str = "rslora") -> float:
    if method == "rslora":
        return alpha / np.sqrt(rank)
    return alpha / rank
