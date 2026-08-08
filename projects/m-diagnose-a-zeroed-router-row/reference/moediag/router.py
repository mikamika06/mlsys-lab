import numpy as np


def find_zeroed_rows(router_weights):
    w = np.asarray(router_weights)
    norms = np.linalg.norm(w, axis=-1)
    zeroed = np.where(norms == 0.0)[0]
    return zeroed.tolist()
