import numpy as np

TEST_CASES = [
    {
        "weight": np.array([[1, 0, 0, 2], [0, 3, 4, 0], [5, 0, 6, 0]], dtype=np.float32),
        "n": 2,
        "m": 4,
        "dim": 1,
        "valid": True
    },
    {
        "weight": np.array([[1, 1, 0, 2], [0, 3, 4, 0], [5, 0, 6, 0]], dtype=np.float32),
        "n": 2,
        "m": 4,
        "dim": 1,
        "valid": False
    },
    {
        "weight": np.zeros((4, 8), dtype=np.float32),
        "n": 2,
        "m": 4,
        "dim": 1,
        "valid": True
    }
]
