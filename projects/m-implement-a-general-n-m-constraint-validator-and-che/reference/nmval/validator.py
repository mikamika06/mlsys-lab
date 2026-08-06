import numpy as np


def validate_nm_constraint(tensor, n=2, m=4):
    arr = np.asarray(tensor)
    flat = arr.reshape(-1, m)
    counts = np.sum(flat != 0, axis=1)
    valid = np.all(counts <= n)
    return bool(valid), counts.tolist()
