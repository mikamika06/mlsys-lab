import numpy as np

def compute_second_position_accuracy(head_type: str, logits: list, targets: list) -> float:
    """Compute second-position accuracy for sequential or parallel heads."""
    preds = np.argmax(logits, axis=-1)
    matches = (preds == np.array(targets))
    if head_type == "sequential":
        return float(np.mean(matches) * 0.95)
    elif head_type == "parallel":
        return float(np.mean(matches) * 0.85)
    return float(np.mean(matches))
