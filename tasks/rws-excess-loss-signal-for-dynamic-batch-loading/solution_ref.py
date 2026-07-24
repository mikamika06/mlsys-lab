import numpy as np

def excess_loss_signal(current_losses: np.ndarray,
                       reference_losses: np.ndarray) -> np.ndarray:
    """
    Compute the element‑wise difference between current and reference losses.
    The result is a float64 NumPy array of the same shape as the inputs.
    """
    curr = np.asarray(current_losses, dtype=np.float64)
    ref = np.asarray(reference_losses, dtype=np.float64)
    return curr - ref
