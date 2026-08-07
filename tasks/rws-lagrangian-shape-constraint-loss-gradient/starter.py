import math

def shape_constraint_loss(logits: list[float], target_sparsity: float, lam: float) -> tuple[float, list[float]]:
    """Compute Lagrangian sparsity loss and gradient."""
    raise NotImplementedError('your code here')
