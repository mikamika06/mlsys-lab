import numpy as np


def get_oracle_data():
    np.random.seed(42)
    raw = np.random.normal(30, 5, 20).tolist()
    return raw
