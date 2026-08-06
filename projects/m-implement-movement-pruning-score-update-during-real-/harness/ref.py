import numpy as np


def get_test_cases():
    np.random.seed(42)
    scores = np.random.randn(8, 8).astype(np.float32)
    weights = np.random.randn(8, 8).astype(np.float32)
    grads = np.random.randn(8, 8).astype(np.float32)
    lr = 0.01

    weight_series = [np.random.randn(8, 8).astype(np.float32) for _ in range(5)]
    grad_series = [np.random.randn(8, 8).astype(np.float32) for _ in range(5)]

    movement_mask = np.random.choice([True, False], size=(8, 8))
    magnitude_mask = np.random.choice([True, False], size=(8, 8))

    return {
        "scores": scores,
        "weights": weights,
        "grads": grads,
        "lr": lr,
        "weight_series": weight_series,
        "grad_series": grad_series,
        "movement_mask": movement_mask,
        "magnitude_mask": magnitude_mask
    }
