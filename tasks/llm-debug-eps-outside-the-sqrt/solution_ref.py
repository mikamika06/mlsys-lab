def layer_norm(x: list[float], gamma: list[float], beta: list[float], eps: float = 1e-5) -> list[float]:
    """LayerNorm with eps correctly placed INSIDE the sqrt."""
    mu = sum(x) / len(x)
    var = sum((xi - mu) ** 2 for xi in x) / len(x)
    std = (var + eps) ** 0.5      # eps inside the sqrt
    return [g * (xi - mu) / std + b for xi, g, b in zip(x, gamma, beta)]
