import numpy as np


def apply_repetition_penalty(logits: np.ndarray, penalty: float) -> np.ndarray:
    # TODO: incorrect implementation divides every logit by the penalty.
    # Negative logits must become more negative by multiplication.
    x = np.asarray(logits, dtype=np.float64)
    return x / penalty
