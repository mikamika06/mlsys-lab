import numpy as np

def optimal_interval(n_layers):
    return int(np.round(np.sqrt(n_layers)))
