import numpy as np


def assign_mixed_precision(model, sensitivity_threshold=0.5):
    assignments = []
    for i, w in enumerate(model.layers):
        sens = float(np.std(w))
        bits = 4 if sens < sensitivity_threshold else 8
        assignments.append((i, bits))
    return assignments
