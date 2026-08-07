import numpy as np

def evaluate_head2(preds, targets):
    """Compare head-2 accuracy to published figures."""
    return float(np.mean(preds == targets))
