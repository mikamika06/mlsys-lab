import numpy as np
from compression.ops import prune, quantize

def measure_both_orders(w: np.ndarray, p: float, b: int):
    """Return a tuple of (prune-then-quantize_result, quantize-then-prune_result)."""
    raise NotImplementedError

def find_interaction_flaw(w: np.ndarray, p: float, b: int):
    """Return a dict with keys 'mse_pq' and 'mse_qp'."""
    raise NotImplementedError

def joint_recipe(w: np.ndarray, p: float, b: int) -> np.ndarray:
    """Prune, but calculate quantization bounds only on the non-pruned elements."""
    raise NotImplementedError

def measure_gains(w_orig: np.ndarray, w_comp: np.ndarray, b: int):
    """Return a dict with 'size_bits' and 'speedup_factor' (total_weights / non_zero_weights)."""
    raise NotImplementedError

def justify_best_order(w: np.ndarray, p: float, b: int):
    """Return a dict with 'best_method': 'joint', 'mse_joint', and 'improvement_over_pq'."""
    raise NotImplementedError
