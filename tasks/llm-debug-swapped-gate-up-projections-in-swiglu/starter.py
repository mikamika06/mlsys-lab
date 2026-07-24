import numpy as np

def _silu(x):
    """Incorrectly applies silu to the gate branch."""
    return x / (1.0 + np.exp(-x))

def swiglu(
    X: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    b_gate: np.ndarray | None = None,
    b_up: np.ndarray | None = None
) -> np.ndarray:
    """
    Buggy implementation: applies silu to the gate projection instead of the up projection.
    """
    gate = X @ W_gate + (b_gate if b_gate is not None else 0.0)
    up   = X @ W_up   + (b_up   if b_up   is not None else 0.0)

    # Wrong: silu applied to gate, then multiplied by up
    return _silu(gate) * up
